# Where this stopped (2026-11-10)

Everything below is uncommitted in the working tree.

## Done

- Listens on `0.0.0.0:8000` (`docker-compose.yml`); API key is in `.env` (git-ignored).
- PDFs are sent to Claude to read (scans, tables, layout), in 20-page pieces read in parallel
  (`llm.transcribe_pdf`, `extract.pdf_batches`). Tesseract, pypdf and pytesseract are gone.
  `PDF_EXTRACTOR=local` falls back to the embedded text layer; `EXTRACT_MODEL` picks the model.
- New single-file UI in `app/static/index.html`. The backend API is unchanged.

- 2026-11-11: Local models are now the default (`LLM_BACKEND=ollama`, `app/ollama.py`).
  `qwen3.5:9b` handles chat and metadata; `minicpm-v4.6` reads only scanned PDF pages. There is an `ollama`
  service in compose. `LLM_BACKEND=claude` brings back the Claude path. Fixed broken newline
  strings in `extract.py`. Tests pass (10), run in a Windows venv rather than the container.

- 2026-11-11: Model pickers in the page ("Read with" for uploads, "Answer with" for chat): Auto, any
  pulled Ollama model, or a Claude model. Auto switches to Claude when Ollama fails or the prompt is too
  long, if a key is available. Table view (DataTables, vendored in `app/static/vendor`) with bulk delete
  (`POST /api/documents/delete`). Documents record which model read them (new `model` column, added to
  old databases automatically). 20 tests pass in a container. UI checked with Playwright in light, dark
  and phone layouts; test docs it made were deleted again.

- 2026-11-11: Instructions panel (header button): rules stored in the `instructions` table and appended
  to the chat system prompt (`llm.system_prompt`), for Ollama and Claude. 24 tests pass. A test of 8
  clearly labelled gallon conversions on qwen3.5:9b scored 8/8 with and without a rule, so the
  reported conversion trouble wasn't reproduced; it needs a real failing question + document.
- 2026-11-11: Found it: the KIOTI manual misprints quarts as gallons ("38.0 ℓ / 40.1 U.S.gal"; Chapter 7
  has the correct 10.0). `app/units.py` corrects litre/gallon pairs in excerpts before the model sees
  them (litres trusted; also handles split table rows and a text layer that lost the ℓ). Stored text is
  unchanged. Real passages, qwen3.5:9b, 18 tries: 18 correct (was mostly "40.1 gallons"). 33 tests pass.
  Note: "Chapter 6 Hydraulic System-1.pdf" is in the library twice (one read by qwen, one by Claude).
- 2026-11-12: Plan items 7, 6 and 16. Uploads are kept in `/data/files/<sha256>` (`FILES_DIR`), once per
  content, with hash/path/size/mime on the document row; `GET /api/documents/<id>/file` serves them (PDF and
  text inline, others as downloads) and the library and table show an **Open** link. Deletes remove a file
  once no document uses it; `db.stored_file(id)` is the hook for re-reading later. Same-hash uploads get
  409 and a Replace / Keep both / Skip prompt in the upload row (replace happens only after the new read
  succeeds). New `usage` table logs every model call (Claude `response.usage` incl. cache tokens, Ollama
  `prompt_eval_count`/`eval_count`), priced from `llm.PRICES`; `GET /api/usage` and a **Usage** dialog.
  All schema changes are added on connect; checked on a read-only copy of the live library (14 docs,
  588 chunks: lists, searches, answers, 404 for originals). 52 tests pass. Staging with qwen3.5, minicpm and
  a few Claude Haiku calls ($0.0007), UI in light/dark and at 320-480px. Fixed on the way: the phone
  header squashed its icon buttons, and the drawer closed when a clicked button removed itself.
- 2026-11-12: Review fixes for items 6, 7 and 16. Cache reads are now priced per model (Opus/Sonnet 5.5
  0.05x, Fable 5.2 0.025x, Haiku 0.1x); before, all four were charged 0.1x, which overstated Claude chat cost.
  The schema update runs once per process under a lock: the page's two parallel first requests sometimes
  got a 500 "duplicate column name" (about 4% of new and 9% of old databases in a stress test, now 0 of
  500 each). A copy of a file still being read counts as a duplicate (reload mid-upload, then pick it again);
  the prompt offers Keep both / Skip. Usage logging can't fail a paid answer or a stored upload any more;
  refused Claude chats are logged (and get a JSON 502); deleting a document clears its usage rows' doc_id
  (ids get reused). Also: a failed insert no longer leaves its file behind, CR/LF in an uploaded file name
  no longer breaks its Open link, ids past 64 bits give 404 instead of 500, and Open is a bigger tap target
  on phones. 60 tests pass.
- 2026-11-12: Plan items 5 and 8, and re-reading (item 7). Uploads are now jobs (`jobs` table, `app/jobs.py`):
  `POST /api/upload` stores the file, checks duplicates and answers 202; one worker thread reads jobs in order
  and saves stage, page counts and every page a model read, so Retry or a restart carries on where it stopped
  (jobs left running are re-queued on the first request after a start). The upload list polls `GET /api/jobs`:
  stage text and a progress bar, Added/Replaced, the Auto switch note, errors with Retry, Cancel, and it all
  comes back after a reload. Re-read: the table view's **Re-read with** sends `POST /api/documents/<id>/reread`,
  which replaces text, chunks, metadata and model in place. PDF readers return one text per page, joined with
  `\f`; chunks keep the page breaks (counting them gives a chunk's page later), and search, excerpts and analysis
  read them as paragraph breaks. Claude now reads only scanned pages by default (`CLAUDE_PDF_PAGES`), as a new
  PDF of those pages with their page numbers; a "Claude reads every page" box overrides it. Claude jobs are
  estimated first and wait for Start at `CONFIRM_COST_USD` ($1.00) or more. Usage is logged per call as it
  happens (new `usage.job_id`), linked to the document when the job ends. Also fixed: an unsupported file no
  longer logs a traceback; "Read with" no longer wraps on phones. 81 tests pass. Staging: qwen3.5 + minicpm
  (progress, cancel mid-read, restart mid-read resumed without re-reading pages), Claude Haiku on 2 small PDFs
  ($0.0007; only the scanned page sent), a 60-page Opus job held at $1.35 and cancelled; UI in light, dark, 390px.
- 2026-11-12: Review fixes for items 5, 7 and 8. A re-read (or an upload replacing its duplicates) whose analysis
  fails now fails instead of wiping the document's title, summary and keywords; Retry redoes only the analysis. When
  only Auto's local analysis fails, Claude analyzes the locally read text instead of reading the pages again (and
  the estimate covers just that). Browser keys: dropped for jobs that won't run, and a job insert that rolls back
  can no longer leave its key to the next job given the same id (another user's key could have been spent).
  The worker starts at boot (`app/gunicorn_conf.py`), so interrupted jobs resume without a page load. Unreadable
  files (damaged PDF/DOCX, not text, empty) get a 400 before queueing again. POSTs naming another site's origin get
  403. Also: a confirmation covers the amount shown; models missing from `PRICES` are estimated at the dearest
  price; a mistyped `CLAUDE_PDF_PAGES` falls back to `scanned`; non-object JSON bodies get 400, not 500; `\f` in
  text files is no page break; Retry counts the pages of the reader it starts with, counted in SQL; compose passes
  `MIN_PAGE_CHARS`, `TOP_K` and the `PDF_*` settings. UI: no orphan row when a poll beats the 202, queued rows
  wait still, "its one page", a jobs badge on the phone ☰, and the phone header no longer overflows at 320/390px
  with an instruction count. 94 tests pass. Staging: restart mid-read resumed with no request; Auto on a scan with a
  broken local chat model made 3 minicpm reads and one Haiku analysis ($0.0001); 25/25 browser checks.
- 2026-11-12: Plan items 3 and 4. Long documents are analyzed section by section (`app/sections.py`): text over one
  call's limit (`CLAUDE_ANALYZE_CHARS`; `OLLAMA_ANALYZE_CHARS`, kept within `OLLAMA_NUM_CTX`) is split at page breaks,
  else paragraphs, into sections of about equal size; the same model takes notes on each (topics, values, procedures,
  warnings), merges notes too long for one local call, and turns them into the metadata in the usual format. At most
  `ANALYZE_MAX_SECTIONS` (24) are read; past that it samples evenly and the analysis says so (`partial: 24 of 61
  sections, 39% of the text`). Job rows show "Analyzing section 3 of 12 with …" with a bar, each call is in the usage
  log (`analyze-section`, `analyze-merge`), notes are kept for a retry, and the Claude estimate counts the sections.
  Unit checks: `units.find()` lists what `fix()` corrects (same code), run at upload and re-read over the stored chunks
  and kept on the document (`unit_findings`, tagged with a hash of units.py, so a changed check runs again); a
  **N unit fixes** badge on cards and in the table opens the list (page links for PDFs). The worker checks older
  documents at start, from their chunks. Fixed on the way: `fix()` paired a gallon figure with the next table cell's
  litres, so Chapter 9's correct "8.9 ℓ, 2.4 U.S.gal | 8.3 ℓ" reached models as 2.2 gal plus a misprint note.
  114 tests pass. Staging on a copy of the live library: all 14 checked at start, 4 quarts misprints listed (chapters
  4, 5, 6-1, 6-2), chapter 9 not flagged; a 98K-character manual in 3 sections by qwen3.5:9b (88 s, prompts at most
  9.3K tokens); Haiku in 2 sections plus the final call ($0.0018, estimated $0.0020); sampling with a cap of 2 and a
  re-read cancelled mid-analysis; 39/39 browser checks (light, dark, 390px).
- 2026-11-12: Review fixes for items 3 and 4. Local analysis could overflow the context: sections were sized at 3
  characters a token, but qwen3.5 gives every digit a token (tables run at 1.3 characters a token; on the live
  library "Chapter 2 Engine-2" runs at 2.0), and Ollama silently dropped the start of a section. Sections are now
  sized by a token estimate shaped like qwen's tokenizer (`ollama.tokens`, within 1% on tables and 1-20% over on
  the library's manuals), analysis calls send `truncate: false` and `shift: false` so Ollama refuses an over-long
  prompt instead of cutting it, and a refused section (or a reply that hit the end of the context) is read again in
  halves; Auto's chat switch uses the same estimate. A merge reply that is cut off or not JSON no longer loses the
  whole analysis: its notes stay unmerged and later rounds merge fewer. The estimate counts sections as `split()`
  makes them (pages kept whole), the final call has its own stage again, sampling reads "(sampled from 61)", and
  file names are cut to 200 characters in prompts. Unit checks: a one-page PDF's findings say page 1 (and a later
  check remembers the text is a PDF's); context and printed figures come from the document's own text, not from what
  an earlier rule corrected, and a bracket the pair opened is closed; at most 500 findings are stored, all counted,
  and listings read the count in SQL (a 6.9 MB table with 196,389 mismatches: listing 450 ms -> 1 ms, list 54.8 MB ->
  139 KB). UI: closing a dialog opened from the phone drawer (✕ or Escape) goes back to the drawer, and the table's
  unit badge is 25 px tall on phones. 126 tests pass. Staging on a copy of the live library: the same 4 misprints,
  a 66K-character half-table text in 4 sections (real prompts 8.5-9.7K tokens; 2 sections of about 17K each before),
  page 1 on a one-page PDF, 515 findings capped at 500; 24/24 browser checks (light, dark, 390px).
- 2026-11-12: Plan items 2, 1 and 11. Citations: chunks record the pages they span (`chunks.page_start`/`page_end`,
  from the kept `\f` breaks, `extract.page_ranges`); excerpts reach the model labelled with file and page(s), with a
  `[Page n]` line where a new page starts inside one, and a note asking for (file, p. n) citations that is added only
  when an excerpt has a page (in the system prompt, qwen3.5 gave page-less excerpts made-up pages in 7 of 15 answers;
  this way 0 of 15). `/api/chat` returns `sources` as one object per document (`doc_id`, `filename`, `pages`,
  `excerpts` indexes) and the `excerpts` as the model read them; chips show "file, p. 4" and open the passages with
  **Open page 4** (`/file#page=4`). Search by meaning (`app/vectors.py`): `qwen3-embedding:0.6b` (`OLLAMA_EMBED_MODEL`)
  embeds each chunk in the upload job ("Indexing" stage) into the new `chunk_vectors` table (float32 + model name); a
  question is embedded with an instruction, compared by cosine (numpy, vectors held in memory and reloaded after
  uploads, re-reads and deletes) and blended with FTS5 by reciprocal rank fusion. Meaning matches under
  `SEARCH_MIN_SIMILARITY` (0.35) are dropped, so "hi" gets no passages, and stopwords are left out of keyword search
  (they made every chunk a keyword match). The worker embeds chunks without vectors when idle (the live library's 588
  at the first start); if Ollama or the model is unavailable, uploads and questions go on by keywords and embedding
  pauses for 2 minutes. Embedding calls are in the usage log at $0; `numpy` added. Scope: `doc_ids` on `/api/chat` limits
  both searches and tells the model; the **Ask** chip under the question box opens a list to filter and tick, there is
  **Ask** on cards and table rows and **Ask about N** in the table, and the choice is kept in localStorage. Also: `/api/chat` gives a
  400, not a 500, for a `history` that isn't a list of turns. 143 tests pass. Staging (qwen3.5, minicpm, qwen3-embedding;
  two Haiku answers, $0.0007): on KIOTI-like chapters, keyword search never finds the "Transmission fluid ... 38.0
  litres" passage for "What is the hydraulic oil capacity?", hybrid search does, and qwen3.5 and Haiku cite p. 4;
  a library made by 894b984 was migrated and its 8 chunks embedded at start, cited by file name only, scoped and
  searched by meaning; a missing embedding model left uploads and chat working and the backfill caught up after; 29/29
  browser checks (light, dark, 390, 360 and 320 px).
- 2026-11-12: Review fixes for items 2, 1 and 11. Broad questions got no passages: "What is this document about?"
  scores 0.19-0.30 against any passage, under the 0.35 floor. Asked of chosen documents, a question now gets their
  nearest passages with no floor, filled up with their opening passages when fewer than `TOP_K` are found; asked of all
  documents and matching nothing ("Summarise everything in my library", "hi"), the model is given the documents' titles
  and summaries (`<library>`, up to 6,000 characters) and the answer says so. A new `SEARCH_MEANING_MARGIN` (0.12)
  leaves out meaning-only passages far behind the nearest: on staging it took the workshop guide's parts lists off
  the warranty, rent and grease questions and kept the "38.0 litres" passage for "hydraulic oil capacity" (0.10
  behind; 0.08 would have dropped it). The prompts name no example file: qwen3.5 cited the made-up "Chapter 7
  Brake.pdf" in 4 of 24 brake answers, and 0 of 24 with the page note's example made from the first paged excerpt
  (right page 20 of 24, before 10); page-less excerpts still got no made-up page (0 of 8). Embedding: a failed batch
  is sent a passage per call, a passage that fails alone is set aside for 30 minutes so the backfill goes on with the
  rest, and questions and passages pause separately. The search box keeps common words when they are all there is
  ("will", "IT"). UI: when every chosen document is deleted the chip says **Chosen document deleted** and questions
  wait (the ids are dropped, since SQLite reuses them); **Replace** keeps a chosen document chosen as its new copy; the
  400 refreshes the library; `[Page n]` lines in a passage open that page. `/api/chat` gives 400 for a history turn
  whose role isn't user/assistant or whose content isn't text. 147 tests pass. Staging (qwen3.5, qwen3-embedding, one
  Haiku call, $0.0002): 14/14 API checks and 18/18 browser checks (light, dark, 390 and 320 px).
- 2026-11-12: Plan items 9 and 10. Streaming: `POST /api/chat/stream` sends Server-Sent Events (`meta` with the
  conversation, model, switch note and sources; `delta` pieces; `done` with usage, or `error` with `kept`), with
  `Cache-Control: no-cache` and `X-Accel-Buffering: no`; Claude streams the same request as before (`llm.chat_stream`,
  `text_stream`), Ollama `/api/chat` with `stream: true` (`ollama.chat_stream`). Auto decides by the fits check, and
  switches to Claude when Ollama fails before its first words; a failure after them ends the answer there (kept, saved
  as broken), a refusal drops its words. The page reads the stream with `fetch`, shows the text as it comes with a caret
  and a **Stop** button (it aborts the fetch; the server closes the model's stream and keeps the answer, marked
  stopped), and falls back to `/api/chat` without streaming. Stopped and broken answers are in the usage log with
  estimated output tokens. Saved conversations: new `conversations` and `messages` tables (each question with what the
  model was sent, each answer with its sources, model, usage and status); both chat routes take `conversation_id`
  (a new conversation, titled by the question, without one) and read the history from the database; a client's own
  `history` still works without it. `GET/PATCH/DELETE /api/conversations[/<id>]`. The page has **Chats**: a column
  beside the chat from 1280 px (shown or hidden, remembered), a drawer below that and on phones (with **New chat**),
  rename (✎) and delete (✕ twice); the open conversation survives a reload, and picking one brings back its model and
  documents. Deleting a document takes it out of conversations' documents, and answers citing it no longer link to the
  file (its id may go to a new upload). Also: one error wording for HTTP and streamed errors (`_failure`); header
  buttons no longer shrink (the ◐ was clipped with a long title), and the header fits from 320 px up with an
  instruction count. 170 tests pass. Staging (qwen3.5, qwen3-embedding, Haiku $0.001): 19/19 API checks (31 pieces
  over 0.8 s through gunicorn, Ollama's log shows a stopped answer's generation cancelled), Auto's switch with a local
  model that isn't pulled, a copy of the live library migrated and answered in a saved conversation (7/7), and 49/49
  browser checks plus 30/30 header checks (light, dark, 320-1400 px).
- 2026-11-12: Review fixes for items 9 and 10. Conversation ids are never given out again (`AUTOINCREMENT`; a table
  made by the first version is copied into one on connect), so a tab or phone holding a deleted one gets 404 instead
  of writing into someone's new chat. A question is kept as it is asked, its answer `answering` until it ends, and the
  stream's first event, `start`, names the conversation and the answer: a page reloaded mid-answer shows the question
  and reads the conversation again until the answer is in (another tab too), a restart marks answers it cut off
  failed, and a refusal or failure keeps the question with the reason (empty answers stay out of history) instead of
  dropping a conversation the page already follows. **Stop**, and leaving or reloading the page, tell the server how
  much the page showed (`POST /api/chat/stop`): exactly that is kept, so "Stopped before the answer began" keeps no
  words. `done` carries the title as it is now (renamed while answering). The conversation's documents are saved when
  the question is asked (a delete meanwhile sticks) and follow a **Replace** to the new copy. Auto keeps a long
  conversation on the local model with the latest exchanges that fit, and says how many questions it left out, instead
  of sending every question from about the 7th to Claude. Questions are limited to 20,000 characters (the page says so
  and keeps the text). **New chat** starts on all documents. Unexpected errors no longer send their text (paths) to
  the page. 184 tests pass. Staging (qwen3.5, nothing spent on Claude): 19/19 API checks, a server killed mid-answer
  (3/3), a 7-question conversation at `OLLAMA_NUM_CTX=4096` kept local (5/5), and 18/18 browser checks (reload and Stop
  mid-answer, another tab, rename, a deleted id, light, dark, 390 px).

## To do

- 2026-11-11: NVIDIA Container Toolkit is installed in `vostok`, Docker has been restarted, both models are
  pulled into volume `chatbot_ollama`, and the stack is up. BUT Ollama runs on the CPU: it logs
  "NVIDIA driver too old … compute=6.1 driver=560 required_driver=570 or newer". Update the Windows
  NVIDIA driver to 570+ (580 series still supports Pascal), reboot, then `docker compose restart ollama`
  and check that `docker compose exec ollama ollama ps` shows GPU.

- Look at the new UI in a browser: it has never been checked visually, in light or dark mode, or on a phone.
- Try a real scanned PDF end to end (not done yet: it spends tokens). Tests pass (6) in a throwaway container built from the new image.
- Commit, if wanted.

## Left on the machine

- Container `chatbot-chatbot-1` is stopped, not removed; the library is in volume `chatbot_metabase`.
  Start it again with `docker compose up -d` inside WSL `vostok`.
- Still open to the LAN, from an elevated prompt remove with:
  `netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0`
  `netsh advfirewall firewall delete rule name="Chatbot 8000"`

## Enhancement plan (2026-11-12)

Ordered by priority. Suggested start: 1 and 2, then 5. Item 12 is small and worth doing early.

### Phase 1: Answer quality

- [x] 1. Search by meaning, not just keywords. FTS5 only matches exact words ("hydraulic oil" missed
      the "transmission fluid" passage). Add an embedding model in Ollama on the GPU and blend its
      results with the keyword search.
- [x] 2. Page-numbered citations. Chunks don't record their page, so answers cite only a filename.
      Store the page range per chunk, show "manual.pdf, p. 142", click to see the excerpt.
- [x] 3. Full analysis of long documents. Text over the character limit gets a "partial" analysis.
      Summarise section by section, then combine, so local models cover the whole document.
- [x] 4. Unit checks at upload. `app/units.py` fixes litre/gallon misprints silently at question
      time. Run it at upload too and list the suspected misprints on the document.

### Phase 2: Uploads

- [x] 5. Background uploads with progress. An upload is one request with a 600 s gunicorn timeout;
      a 200 MB scan on the local model can exceed it. Job queue, "page 40 of 312", resume on failure.
- [x] 6. Duplicate detection. Hash the file and offer replace or skip
      ("Chapter 6 Hydraulic System-1.pdf" is in the library twice).
- [x] 7. Keep the original file. Only extracted text is stored. Keeping the PDF allows re-reading
      with a different model without re-uploading, and viewing the cited page.
- [x] 8. Cheaper Claude reading. Send only scanned pages to Claude and use the text layer for the
      rest (the Ollama path already does). Show an estimated cost before a large upload.

### Phase 3: Chat experience

- [x] 9. Streaming answers, instead of the whole reply appearing after a wait.
- [x] 10. Saved conversations. History lives only in the browser tab; store it on the server.
- [x] 11. Scope a question to chosen documents ("ask only this manual").

### Phase 4: Safety and operations

- [ ] 12. A login. The app is open to the LAN on port 8000 with no password; anyone can delete
      documents or change the instructions. A single shared password would close that.
- [ ] 13. Stable network address. The Windows portproxy points at the WSL IP, which can change on
      restart. Use WSL mirrored networking or a start-up script that re-points it.
- [ ] 14. Backups. The library is one SQLite file in volume `chatbot_metabase`. Nightly copy.
- [ ] 15. Run the tests on GitHub (Actions workflow on every push); today they run by hand.
- [x] 16. Usage log. Record tokens and model per request to see what Claude costs.
