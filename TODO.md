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

- [ ] 1. Search by meaning, not just keywords. FTS5 only matches exact words ("hydraulic oil" missed
      the "transmission fluid" passage). Add an embedding model in Ollama on the GPU and blend its
      results with the keyword search.
- [ ] 2. Page-numbered citations. Chunks don't record their page, so answers cite only a filename.
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

- [ ] 9. Streaming answers, instead of the whole reply appearing after a wait.
- [ ] 10. Saved conversations. History lives only in the browser tab; store it on the server.
- [ ] 11. Scope a question to chosen documents ("ask only this manual").

### Phase 4: Safety and operations

- [ ] 12. A login. The app is open to the LAN on port 8000 with no password; anyone can delete
      documents or change the instructions. A single shared password would close that.
- [ ] 13. Stable network address. The Windows portproxy points at the WSL IP, which can change on
      restart. Use WSL mirrored networking or a start-up script that re-points it.
- [ ] 14. Backups. The library is one SQLite file in volume `chatbot_metabase`. Nightly copy.
- [ ] 15. Run the tests on GitHub (Actions workflow on every push); today they run by hand.
- [x] 16. Usage log. Record tokens and model per request to see what Claude costs.
