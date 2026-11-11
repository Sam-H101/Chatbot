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

The database is a single file, `/data/metabase.db`, kept in the `metabase` Docker volume. No database server is needed.

## What it does

| Feature | How |
|---|---|
| Upload | PDF (text or scanned), DOCX, and any UTF-8 text file (txt, md, csv, json, code…) |
| PDF reading | PDFs are sent to Claude, which sees each page as text and as an image. Scans, tables, multi-column layouts and charts all come out as clean Markdown. Long PDFs are split into `PDF_PAGES_PER_CALL`-page pieces that are read in parallel |
| Analyze | One Claude call per document with structured JSON output: title, doc type, summary, keywords |
| Metadata base | SQLite tables `documents` and `chunks`, plus FTS5 indexes over the metadata and the full text (porter stemming) |
| Search | The sidebar search box matches documents by metadata and lists matching passages with highlighted snippets |
| Chat | Each question runs a full-text search for the top 5 passages, which go into the prompt. Answers cite file names |
| Model pickers | "Read with" (in the upload box) picks the model that reads and analyzes uploads; "Answer with" (under the question box) picks the chat model. Each offers **Auto**, every model pulled into Ollama, and the Claude models. The page remembers both choices, and every answer and upload shows which model did the work |
| Auto | Uses the local model (`OLLAMA_MODEL`) and switches to Claude (`CLAUDE_MODEL`) when Ollama is down or errors, or when a chat prompt is too long for `OLLAMA_NUM_CTX`. It only switches when a Claude key is available (browser or server), and says why it switched |
| Instructions | The **Instructions** button holds rules added to the system prompt of every question, for local models and Claude alike (for example how to handle units). Each can be switched on or off, and they are stored in the database, so they apply to everyone using the app |
| Unit checks | Before passages go to the model, litre and gallon figures printed side by side are checked against each other (`app/units.py`). If they disagree, the litre figure is trusted and the gallon figure is recalculated; a quarts-printed-as-gallons misprint is shown as quarts. The stored document text is not changed |
| Table view | The **Table** button in the library opens every document in a sortable, filterable table (DataTables, served from `app/static/vendor`, so it works offline). Tick rows, or the header box to tick everything the current filter shows, then delete them all at once |

## Keeping token use down

- **Retrieval instead of whole documents:** a chat turn sends only the top-K passages (`TOP_K`, default 5, about 1,500 characters each), not entire files.
- **Prompt caching:** chat requests set `cache_control`, so the system prompt and the earlier conversation are read from cache (about 10% of the normal input price). The browser sends back the exact earlier turns so the prefix stays byte-identical. Each answer shows how many tokens were read from and written to the cache.
- **Bounded analysis:** metadata extraction reads at most the first `CLAUDE_ANALYZE_CHARS` (default 500,000) characters of a document with Claude, or `OLLAMA_ANALYZE_CHARS` (default 40,000) with a local model, which has a much smaller context window. The full text is always indexed and searchable. Documents longer than the limit are marked `partial`. A 500,000-character document is roughly 125,000-170,000 input tokens, so a full analysis costs noticeably more on Opus than on Haiku or Sonnet.
- **PDF reading costs tokens:** each page is billed as an image plus its text (roughly 1,500 to 3,000 input tokens), and the transcription is billed as output. Pick Claude Haiku 5.5 under "Read with" to make this cheap, or `PDF_EXTRACTOR=local` to spend nothing on text PDFs.
- **Low effort by default:** `CLAUDE_EFFORT=low`. Raise it if you want more careful answers.

## Configuration

| Env var | Default | |
|---|---|---|
| `LLM_BACKEND` | `ollama` | Default when the page sends no model choice: `ollama` means Auto, `claude` means `CLAUDE_MODEL` |
| `OLLAMA_URL` | `http://ollama:11434` (compose) | Where Ollama listens. Point it elsewhere to use an Ollama server outside compose |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Chat and metadata model |
| `OLLAMA_VISION_MODEL` | `minicpm-v4.6` | Reads scanned PDF pages. `qwen3.5:9b` also reads images, so set it to that to run a single model |
| `OLLAMA_NUM_CTX` | `16384` | Context window per request. About the most an 11 GB card fits next to a 9B model |
| `MIN_PAGE_CHARS` | `20` | Ollama only: a PDF page with less embedded text than this counts as scanned |
| `ANTHROPIC_API_KEY` | (none) | Optional server-side key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model Auto switches to, and the default when `LLM_BACKEND=claude`. Any Claude model can also be picked in the page |
| `CLAUDE_EFFORT` | `low` | `low` / `medium` / `high` / `xhigh` / `max` |
| `SYSTEM_PROMPT` | built-in | Gives the bot its personality |
| `CLAUDE_ANALYZE_CHARS` | `500000` | Characters per document Claude reads for metadata extraction |
| `OLLAMA_ANALYZE_CHARS` | `40000` | The same for local models. Keep it within `OLLAMA_NUM_CTX` (about 3 characters per token), or Ollama silently cuts the text |
| `TOP_K` | `5` | Passages retrieved for each question |
| `PDF_EXTRACTOR` | `model` | `model` lets the backend's model read PDFs (with Ollama, only scanned pages). `local` uses only the PDF's embedded text layer, so scanned pages come out empty |
| `PDF_PAGES_PER_CALL` | `20` | Pages per Claude call when reading a PDF |
| `PDF_CONCURRENCY` | `4` | PDF pieces read at the same time |
| `MAX_UPLOAD_MB` | `200` | Maximum upload size. The page checks it before uploading |
| `PDF_MAX_BATCH_MB` | `20` | Largest PDF piece sent to Claude in one request (about 27 MB once base64-encoded, under the API's 32 MB limit). Pieces over it are split further |
| `DB_PATH` | `/data/metabase.db` | Location of the SQLite file |

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
