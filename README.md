# Document Chatbot (PoC)

A small standalone chatbot that answers questions with the Claude API, using your own API key. You upload documents, Claude pulls metadata out of each one (title, type, summary, keywords), and everything goes into an embedded SQLite database with full-text search. When you ask a question, the app finds the most relevant passages and sends them along with it.

## Run it

By default the app runs on local models through [Ollama](https://ollama.com), on your own NVIDIA GPU: no API key, and documents never leave the machine. The defaults are sized for an 11 GB card (GTX 1080 Ti):

- `qwen3.5:9b` (about 6.6 GB) answers questions and extracts metadata.
- `minicpm-v4.6` (about 1.6 GB) reads scanned PDF pages. Pages that have embedded text skip the model.

Containers need the NVIDIA Container Toolkit to use the GPU (one-time setup, inside WSL on Windows). See NVIDIA's install guide, then run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen3.5:9b
docker compose exec ollama ollama pull minicpm-v4.6
```

Open http://localhost:8000 and upload some files.

To use the Claude API instead, run `LLM_BACKEND=claude docker compose up --build` and paste your Claude API key into the page. The rest of this README describes both backends. Token and caching notes apply to Claude.

The key stays in your browser's localStorage and goes along with each request. The server never stores it. If you'd rather set it on the server, export `ANTHROPIC_API_KEY` before running `docker compose up`.

The database is a single file, `/data/metabase.db`, kept in the `metabase` Docker volume. No database server is needed. The original uploaded files are kept beside it, in `/data/files`. Uploads are read in the background by one worker thread, which is why gunicorn runs a single worker process (see the Dockerfile); its queue lives in the database.

## What it does

| Feature | How |
|---|---|
| Upload | PDF (text or scanned), DOCX, and any UTF-8 text file (txt, md, csv, json, code…). The upload only sends and stores the file (with a progress bar while it travels); reading happens in the background, see the next row |
| Background reading | Each upload becomes a job in the `jobs` table, and one worker thread reads the jobs in order. The upload list shows each job live: its stage ("Reading page 40 of 312", "Analyzing with …") and a progress bar, then **Added · model** (or the reason Auto switched to Claude). Jobs survive a page reload: the list comes back from `GET /api/jobs`. A running job can be cancelled (it stops after its current page). A failed job shows its error with **Retry**; pages a model already read are kept with the job, so a retry, or a job cut off by a restart, carries on where it stopped and doesn't pay for those pages twice. Jobs left running by a restart are queued again when the app next gets a request. API: `POST /api/upload` answers 202 `{job}`; `GET /api/jobs`, `GET /api/jobs/<id>`, `POST /api/jobs/<id>/cancel`, `/retry` and `/confirm` |
| PDF reading | Every reader returns one text per page, and a document's text keeps its pages apart with a form feed (`\f`), so a chunk's page can be worked out later; search, excerpts and analysis treat it as a paragraph break. Local models: pages with a text layer are taken as they are, and the vision model (`OLLAMA_VISION_MODEL`) reads only scanned pages. Claude: by default (`CLAUDE_PDF_PAGES=scanned`) the same: only scanned pages go to Claude, as a new PDF of just those pages, with their page numbers in the prompt. The **Claude reads every page** box beside "Read with" (shown for Claude models) has Claude read every page as text and image instead, which is worth its cost for complex tables and layouts. Claude reads `PDF_PAGES_PER_CALL`-page pieces in parallel. A page counts as scanned when its text layer has fewer than `MIN_PAGE_CHARS` visible characters |
| Cost estimate | Before a job reads with Claude, it estimates the cost: pages, pages Claude reads, about 2,000 input and 600 output tokens per page Claude reads, plus the metadata call (about 4 characters a token), priced with `PRICES` in `app/llm.py`. The upload row shows the estimate. At `CONFIRM_COST_USD` (default $1.00) or more, the job waits for **Start** or **Cancel**. Auto only estimates Claude when it actually switches to it |
| Re-read | In the **Table** view, tick documents and pick **Re-read with** (any model, with the same every-page box for Claude): their kept originals are read again and each document's text, metadata and model are replaced in place, keeping its id and upload date. Documents uploaded before originals were kept can't be re-read. API: `POST /api/documents/<id>/reread` with `{"model": "…", "claude_pages": "scanned" \| "all"}` |
| Analyze | One call per document with structured JSON output: title, doc type, summary, keywords |
| Metadata base | SQLite tables `documents` and `chunks`, plus FTS5 indexes over the metadata and the full text (porter stemming), and `jobs`, `usage` and `instructions`. Older databases gain new columns and tables automatically when the app starts; their documents keep working without a kept file, hash or page breaks |
| Originals | Every uploaded file is kept in `FILES_DIR` (default `/data/files`), stored once per distinct content and named by its SHA-256. **Open** in the library and in the table view shows it: PDFs open in the browser's viewer and plain text as text; other types download. `GET /api/documents/<id>/file` serves it. Deleting a document removes its file once no other document uses it. Documents uploaded before originals were kept have no file and no Open link |
| Duplicates | Uploads are hashed (SHA-256). If the same file is already in the library, the upload stops before any model reads it, and its row in the upload queue asks: **Replace** (read the new upload, then delete the earlier copies), **Keep both**, or **Skip**. Through the API: the form field `on_duplicate` is `replace` or `keep`; without it the answer is 409 with the earlier copy's id, file name, date and model. A copy that is still queued or being read also counts (say the file was picked twice): the 409 then has `in_progress: true`, and only **Keep both** or **Skip** are offered. Documents uploaded before hashing never count as duplicates |
| Usage | Every model call (reading PDF pages, analyzing a document, answering) is logged with its model, tokens and cost in the `usage` table as it happens, including calls made by uploads that then failed and Claude answers that ended in a refusal. Calls made by a job carry its `job_id` and are linked to the document when the job finishes (failed attempts before a successful retry too). Deleting a document keeps its rows but clears their `doc_id`, because SQLite reuses ids. The **Usage** button shows totals per model and per day for the last 7, 30, 90 or 365 days; `GET /api/usage?days=30` returns the same. Claude costs come from the list prices in `PRICES` in `app/llm.py` and are estimates, not the invoice. Cache reads have their own price per model (5% of the input price on Opus and Sonnet 5.5, 2.5% on Fable 5.2, 10% on Haiku 5.5), and cache writes cost 1.25x the input price. Models missing from the table count as $0. Local models cost nothing. If the log can't be written (say the database stays locked), the answer or upload still goes through and the error goes to the server log |
| Search | The sidebar search box matches documents by metadata and lists matching passages with highlighted snippets |
| Chat | Each question runs a full-text search for the top 5 passages, which go into the prompt. Answers cite file names |
| Model pickers | "Read with" (in the upload box) picks the model that reads and analyzes uploads; "Answer with" (under the question box) picks the chat model. Each offers **Auto**, every model pulled into Ollama, and the Claude models. The page remembers both choices, and every answer and upload shows which model did the work |
| Auto | Uses the local model (`OLLAMA_MODEL`) and switches to Claude (`CLAUDE_MODEL`) when Ollama is down or errors, or when a chat prompt is too long for `OLLAMA_NUM_CTX`. It only switches when a Claude key is available (browser or server), and says why it switched |
| Instructions | The **Instructions** button holds rules added to the system prompt of every question, for local models and Claude alike (for example how to handle units). Each can be switched on or off, and they are stored in the database, so they apply to everyone using the app |
| Unit checks | Before passages go to the model, litre and gallon figures printed side by side are checked against each other (`app/units.py`). If they disagree, the litre figure is trusted and the gallon figure is recalculated; a quarts-printed-as-gallons misprint is shown as quarts. The stored document text is not changed |
| Table view | The **Table** button in the library opens every document in a sortable, filterable table (DataTables, served from `app/static/vendor`, so it works offline). Tick rows, or the header box to tick everything the current filter shows, then delete them all at once |

## Keeping token use down

- **Retrieval instead of whole documents:** a chat turn sends only the top-K passages (`TOP_K`, default 5, about 1,500 characters each), not entire files.
- **Prompt caching:** chat requests set `cache_control`, so the system prompt and the earlier conversation are read from cache (5% of the normal input price on Opus and Sonnet 5.5, 2.5% on Fable 5.2, 10% on Haiku 5.5). The browser sends back the exact earlier turns so the prefix stays byte-identical. Each answer shows how many tokens were read from and written to the cache.
- **Bounded analysis:** metadata extraction reads at most the first `CLAUDE_ANALYZE_CHARS` (default 500,000) characters of a document with Claude, or `OLLAMA_ANALYZE_CHARS` (default 40,000) with a local model, which has a much smaller context window. The full text is always indexed and searchable. Documents longer than the limit are marked `partial`. A 500,000-character document is roughly 125,000-170,000 input tokens, so a full analysis costs noticeably more on Opus than on Haiku or Sonnet.
- **PDF reading costs tokens:** each page Claude reads is billed as an image plus its text (roughly 1,500 to 3,000 input tokens), and the transcription is billed as output. By default Claude reads only scanned pages, so a PDF with a text layer costs just its metadata call. Leave **Claude reads every page** off unless the layout needs it, pick Claude Haiku 5.5 under "Read with" to make reading cheap, or use a local model to spend nothing. Jobs estimated at `CONFIRM_COST_USD` or more wait for a confirmation.
- **Low effort by default:** `CLAUDE_EFFORT=low`. Raise it if you want more careful answers.

## Configuration

| Env var | Default | |
|---|---|---|
| `LLM_BACKEND` | `ollama` | Default when the page sends no model choice: `ollama` means Auto, `claude` means `CLAUDE_MODEL` |
| `OLLAMA_URL` | `http://ollama:11434` (compose) | Where Ollama listens. Point it elsewhere to use an Ollama server outside compose |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Chat and metadata model |
| `OLLAMA_VISION_MODEL` | `minicpm-v4.6` | Reads scanned PDF pages. `qwen3.5:9b` also reads images, so set it to that to run a single model |
| `OLLAMA_NUM_CTX` | `16384` | Context window per request. About the most an 11 GB card fits next to a 9B model |
| `MIN_PAGE_CHARS` | `20` | A PDF page with less embedded text than this counts as scanned: the vision model reads it (Ollama), and so does Claude when `CLAUDE_PDF_PAGES=scanned` |
| `CLAUDE_PDF_PAGES` | `scanned` | Which PDF pages Claude reads: `scanned` (only pages without a usable text layer; the text layer is used for the rest) or `all` (every page, as text and image). The default for the page's **Claude reads every page** box; uploads can send `claude_pages` to override it |
| `CONFIRM_COST_USD` | `1.00` | A job whose Claude reading is estimated to cost this much or more waits for **Start** in the upload list. `0` asks for every Claude job |
| `ANTHROPIC_API_KEY` | (none) | Optional server-side key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model Auto switches to, and the default when `LLM_BACKEND=claude`. Any Claude model can also be picked in the page |
| `CLAUDE_EFFORT` | `low` | `low` / `medium` / `high` / `xhigh` / `max` |
| `SYSTEM_PROMPT` | built-in | Gives the bot its personality |
| `CLAUDE_ANALYZE_CHARS` | `500000` | Characters per document Claude reads for metadata extraction |
| `OLLAMA_ANALYZE_CHARS` | `40000` | The same for local models. Keep it within `OLLAMA_NUM_CTX` (about 3 characters per token), or Ollama silently cuts the text |
| `TOP_K` | `5` | Passages retrieved for each question |
| `PDF_EXTRACTOR` | `model` | `model` lets the chosen model read PDFs (scanned pages only, unless Claude is told to read every page). `local` uses only the PDF's embedded text layer, so scanned pages come out empty |
| `PDF_PAGES_PER_CALL` | `20` | Pages per Claude call when reading a PDF |
| `PDF_CONCURRENCY` | `4` | PDF pieces read at the same time |
| `MAX_UPLOAD_MB` | `200` | Maximum upload size. The page checks it before uploading |
| `PDF_MAX_BATCH_MB` | `20` | Largest PDF piece sent to Claude in one request (about 27 MB once base64-encoded, under the API's 32 MB limit). Pieces over it are split further |
| `DB_PATH` | `/data/metabase.db` | Location of the SQLite file |
| `FILES_DIR` | `files` next to `DB_PATH` (`/data/files`) | Where the original uploads are kept, one file per distinct content, named by its SHA-256 |

Requests turn on server-side refusal fallbacks (`fallbacks: "default"`). If Claude's safety filter declines a request, the API retries it on a fallback model instead of failing.

## Develop without Docker

```bash
pip install -r requirements.txt pytest
DB_PATH=data/metabase.db flask --app app.main run --debug
pytest
```

## Not done yet

- Batch analysis for bulk uploads (the Message Batches API is 50% cheaper but asynchronous)
- Semantic or embedding search (FTS5 matches keywords, not meaning)
- Authentication (anyone who can reach the port can see the documents)
