# Where this stopped (2026-11-10)

Everything below is committed (as of 2026-11-12, the Documents view included); nothing after 894b984 has been pushed yet.

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
- 2026-11-12: Plan item 12. An optional shared password (`app/auth.py`). With `APP_PASSWORD` set, every page and API
  call needs a signed session cookie (`docchat_session`: HttpOnly, SameSite=Lax, `SESSION_DAYS` = 30 days from
  logging in, not renewed by every poll) from `/login`, a standalone page (inline styles, light and dark, phone
  width). Pages redirect there and back after it (a bookmarked link to an original file too); `/api/*` gets 401
  `{"error": "Login required", "login": true}` and the page then goes to the login page without an error box.
  Passwords are compared with `hmac.compare_digest` over SHA-256 digests; the session holds an HMAC of the password,
  so a new password logs everyone out. The cookie key is `SECRET_KEY`, or else a random one made once in
  `/data/secret_key` (mode 600), loaded by a session interface when first needed (Flask opens the session before any
  `before_request` hook). Five wrong passwords from one address within 10 minutes make it wait a minute after each
  further failure (429 with `Retry-After`, the right password included). CSRF: logged in, POST/PATCH/DELETE under
  `/api/` need an `X-Requested-With` header, which the page's calls, the XHR upload, the stream and the stop request
  send; the Origin check stays. **Log out** in the header (an icon at 420 px and below, where Instructions reads
  **Rules** to make room; it fits from 320 px). Without a password nothing changes: no cookie, no header needed, no
  key file, the header as before. Compose passes `APP_PASSWORD`, `SECRET_KEY` and `SESSION_DAYS` (no defaults).
  193 tests pass. Staging (qwen3.5, nothing spent on Claude): 28/28 API checks (an upload read and embedded and a
  streamed answer while logged in, a job finished after a restart with no request at all, the session kept across
  the restart, the throttle) and 52/52 browser checks (light, dark, 1400-320 px, with and without a password).
- 2026-11-12: Review fixes for item 12. The header overflowed with a login at 421-446, 481-492 and 821-893 px (large
  iPhones at 428-440, an 11" iPad at 834; Log out cut off) and, without one, at 341-364 px with a two-digit
  instruction count (from before item 12): short labels now apply with a login from 500 px down and at 821-910 px
  (Key there without one too), and the 380/350 px tightening applies to both; measured at every width from 320 to
  1000 px plus 1099-1400, with 0, 1 and 12 instructions, light and dark, with and without a password: 0 overflows.
  A call answered 401 sends the page to `/login?ended=1`, which says the session ended; a question being sent (until
  the server kept it) or typed, and a new instruction, come back in their boxes after logging in (sessionStorage).
  **Log out** reports a streamed answer's stop before leaving (it got 401 on pagehide, so more was kept than shown),
  sends it once, and forgets the Claude key typed into that browser. The login page adds its `#page=5` to `next`,
  so a page link keeps its page. The log says at start whether a login is on. README: `.env` value in single quotes
  (`$` and ` #` were mangled; `$ecret1` became no password at all), `curl --data-urlencode`, the key file is made at
  the first request, a wrong try a minute keeps the lock going for everyone and a restart clears it. 196 tests pass.
  Staging (qwen3.5, nothing spent on Claude): drafts 11/11, page link 4/4, stream stopped by Log out, reload and
  Stop kept exactly as shown (one 204 each), login page 6/6, curl flow with `pa$s #w+r&d`, `.env` quoting checked
  with `docker compose config`.
- 2026-11-12: Second review fixes for item 12. A Log out whose POST got no answer (stopped, or the server
  unreachable) left the page set as leaving, so a later 401 no longer sent it to the login page and the call just
  hung; the page now goes there again after 10 seconds, or at once when Back brings it from the back/forward cache.
  A three-digit instruction count left the last header button 3-5 px out at 320-324 and 351-355 px: the 350 px
  tightening now starts at 360 px, and buttons lose another pixel each side at 330 px and below. 196 tests pass.
  Staging (nothing spent on Claude): logout 9/9 plus 5/5 for Back, header fit at every width from 320 to 1000 px
  plus 1099-1400 with 1, 12, 99, 100 and 999 instructions, with and without a login, phone and desktop: 0 overflows.
- 2026-11-12: Third review fix for item 12. A second Log out within 10 s of a first whose POST got no answer had its
  wait cut short by the first one's timer; the page now keeps the time of the latest Log out instead of a timer, so
  each gets its full 10 s. 196 tests pass. Staging (nothing spent on Claude): logout 15/15, including a second Log
  out 7 s after the first holding until 10 s after the second, and 5/5 for Back.
- 2026-11-12: Plan items 13, 14 and 15. Backups (`app/backup.py`): a `backup` compose service (the same image) snapshots
  `/data/metabase.db` every night at `BACKUP_HOUR` (3, the container's time: UTC unless `TZ` is set) with SQLite's
  backup API, from the library mounted read-only and opened with `mode=ro`; each copy is compacted (`VACUUM`) and
  checked before it is named `metabase-YYYYMMDD-HHMM.db` in the `backups` volume, the newest `BACKUP_KEEP` (14) are
  kept, and new originals are added to `files/` there (never pruned, checked against their SHA-256). A night the
  machine was off or asleep is made up when it is back; a failure is tried again 30 minutes later. `--now` and
  `--list`, and a `restore` service behind a compose profile (`docker compose run --rm restore <snapshot>|latest`):
  it checks the snapshot, keeps the database it replaces as `.before-restore-<time>`, and copies back the originals
  its documents use. Network: `scripts/portproxy.ps1` starts WSL and keeps it running (a hidden `wsl.exe --exec sleep
  infinity`), reads its address and adds or updates the `0.0.0.0:8000` port proxy rule, starts IP Helper, and adds the
  firewall rule if it is missing; running it again changes nothing, it logs to `C:\ProgramData\Chatbot\portproxy.log`,
  `-DryRun`/`-WhatIf` only print, and without elevation it refuses to change anything. `scripts/install-portproxy-task.ps1`
  copies it to an admin-only `C:\ProgramData\Chatbot` and registers the task "Chatbot portproxy" (as you, highest
  privileges, at startup without a stored password and at logon; `-Interactive`, `-Remove`). Tests on GitHub:
  `.github/workflows/tests.yml` (Python 3.13, `requirements.txt` plus pytest and pillow, `python -m pytest -q`; plain
  `pytest` can't import the app, so the README now says `python -m pytest`). 207 tests pass, also in a fresh
  python:3.13-slim as a non-root user with the network off. Staging: the backup service on the staging library (first
  snapshot at start, `/data` read-only in it, 11 snapshots taken while qwen3.5 read and embedded an upload, all
  checked, stopped in 0.3 s); a restore through the compose service brought back two deleted documents and their
  originals, and qwen3.5 answered from them with page citations; a read-only snapshot of the live library came to 8.1 MB
  from 97 MB (14 documents, 588 chunks, integrity ok, the live file untouched). The PowerShell scripts parse, their dry
  runs and refusals were checked under a non-admin token, and the task has not been installed. Also: the stale "To
  do" and "Left on the machine" notes above are brought up to date.
- 2026-11-12: Review fixes for items 13 and 14. A restore no longer swaps the database under a running app (which
  kept the old library's vectors, schema and job ids in memory, so a question could get a deleted document's
  passage): gunicorn's master holds a shared flock on `/data/app.lock` while the app runs, the restore takes it
  exclusively or refuses with "stop it first", and an app starting meanwhile waits for it; it then says
  `docker compose restart chatbot`, which is right even if an app without the lock was left running. The `restore`
  service is built at every run (`pull_policy: build`): `up --build` skips it, so it kept the code of its first
  build. The install script took over an existing `C:\ProgramData\Chatbot` only in name: an earlier owner or
  explicit permissions survived, so a standard user who made the folder first could change the script the task
  runs with full rights. Now Administrators take ownership (`takeown /A`), its permissions are reset to the three
  entries, a script already there is replaced by a new file, a link in its place is refused, and the install stops
  unless owner and permissions check out. portproxy.ps1 logged "Started" and exited 0 when IP Helper or the
  keepalive failed to start; it now logs the failure, carries on, and exits 1 (a keepalive that ends at once counts
  too). README: copy backups to another disk, not into the repo (`backups/` is now git-ignored); a mistyped
  `BACKUP_HOUR` makes Docker restart the service, it doesn't stop it. "To do" notes that a deploy naming only
  `chatbot` would never create `backup`. 208 tests pass, also in the CI mimic. Staging: a
  restore refused while staging ran (nothing moved) and worked once it stopped; staging waited 12 s for a held lock;
  after a code change `run restore` used the new code. The PowerShell checks ran on scratch folders and with mocked
  service and process starts (nothing on the machine changed); the task is still not installed.
- 2026-11-12: Fixes from the integration review, which ran the new code on a copy of the live library. Local
  models: analysis calls no longer send `shift: false`. Ollama starts a model with that setting, so qwen3.5 was
  restarted between every analysis call and question (136 starts in Ollama's log that day, 5-13 s each), and after a
  restart mid-answer such calls went unanswered, so uploads failed after 10 minutes. A reply that runs past the
  context is now caught from its token counts. A call Ollama takes but never answers fails with "no answer within
  600 s (OLLAMA_TIMEOUT); Ollama may be busy", not "Could not reach Ollama ... Is it running?". **Cancel**, or
  deleting a document being re-read, cuts off the job's Ollama call at once: on staging the next upload started 0.1 s
  after Cancel, and Ollama logged "cancel task". The pickers no longer offer the embedding model (`/api/show`
  capabilities). Answers: the system prompt no longer invites general-knowledge figures. qwen3.5 had given the
  tractor a made-up 11.0 L hydraulic oil capacity; asked of a chapter without tire pressures, it gave pressures from
  general knowledge in 2 of 3 answers before and 0 of 3 now, while still explaining terms ("SAE 80/90").
  Search by meaning: scraps of a longer document under 100 characters are left out (a 57-character table scrap was
  nearest to nearly every short question, "hi" included), and greetings are stopwords. `SEARCH_MIN_SIMILARITY` is now
  0.47, measured on the live library: small talk scored 0.45 at most, and real questions' nearest passages 0.59-0.81.
  Now 17 of 24 small-talk and library questions get the overview (1 before). All 23 real questions still get 5
  passages: 18 are unchanged, and the 3 that had the scrap get a real passage in its place. Database: a 30 s busy timeout (saving a 44 MB upload held the database for 7 s, past the old 5 s). A job
  whose failure can't be written is retried, rather than left 'running'. Deleting a document drops the pages its
  re-read kept. Vectors load into arrays sized up front: the peak for 50,000 chunks fell from 625 MB to 206 MB. UI:
  the key dot is green when the server has a key, and phone drawer cards no longer overflow at 320-359 px. Ops:
  gunicorn's `--graceful-timeout 8` means a restart with an answer streaming stops cleanly in 8.5 s, inside Docker's
  10 s. Compose passes `OLLAMA_TIMEOUT`, `OLLAMA_KEEP_ALIVE` and `RENDER_SCALE`, now in the README, which also says
  `DB_PATH`, `FILES_DIR` and `BACKUP_DIR` are fixed by the image. 216 tests pass. Staging on a copy of the live library
  (qwen3.5, nothing spent on Claude): 10/10 end-to-end checks (cancel mid-analysis, an upload then a question then an
  upload on one runner, restart mid-answer then an upload) and 19/19 browser checks (light, dark, 320-390 px drawer).
  The deploy script now tags a rollback image (`chatbot-chatbot:894b984`, tagged already), copies the library first,
  logs in when a password is set, and deletes its smoke-test conversation.
- 2026-11-12: Deployed the enhancement plan to the live app (code of c495ea6; 216 tests passed first). Before it, two
  copies of the library made with SQLite's backup API: `/data/metabase-before-enhancements-20261112.db` in the volume
  (97 MB; 14 documents, 588 chunks, 1 instruction, integrity ok, the same counts as the live file) and the deploy
  script's `~/chatbot-predeploy/metabase-20261112-195436.db` in WSL; the old code is still tagged `chatbot-chatbot:894b984`.
  `docker compose up -d --build chatbot backup`: the library gained the new tables and columns at the first start, all
  14 documents were unit-checked (chapters 4, 5, 6-1 and 6-2 show **1 unit fix** each, the quarts misprint), and all 588
  passages were embedded by the worker within about two minutes. No password is set (`APP_PASSWORD` unset), so there
  is no login. Live checks, changing nothing of the library: `/api/models` (Auto, qwen3.5:9b, no embedding model in the
  pickers); the 14 documents with their ids and names, no Open link or original (404); keyword search for "hydraulic oil"
  (10 hits); a streamed Auto answer (first words 2.9 s, 671 pieces, qwen3.5:9b with no switch) whose five passages were
  all found by both keywords and meaning and which cited the DK20 manual and "Chapter 6 Hydraulic System-2.pdf" by name
  (the live documents have no page numbers); its conversation saved whole, then deleted; `/api/usage` (400 for `days=0`);
  an upload of `zz-deploy-check-20261112.txt` with qwen3.5:9b (done in 3 s, embedded, Open gave it back byte for byte,
  the same file again got 409), then deleted with its kept file. The checks' delete steps hit a bug in the check script
  after the requests had gone out; a second run confirmed the conversation and the document gone and `/data/files`
  empty. Backups: the first snapshot was taken at start (8.1 MB) and `--now` made a second (10.8 MB, integrity ok, 588
  vectors), in the new `chatbot_backups` volume. Browser (Playwright on `localhost:8000`): 42/42 in light and dark at
  1400, 390 and 320 px: library cards with 4 unit badges and the unit dialog, the table view with 14 rows and Re-read
  disabled, Usage, Chats, the Ask dialog ("searched by keywords and by meaning"), a question streamed in the page with
  its source chips and passage dialog (that conversation deleted again), no console or page errors, no sideways scroll.
  From Windows, `http://192.168.112.184:8000` answers (page, search status, 14 documents), and a DELETE and a POST
  naming that address as their origin pass the other-site check (404 for ids that don't exist; another origin gets 403).
  The checks left 32 rows in the usage log (embedding and qwen3.5 calls, all local, $0) and the test upload's finished
  job row. No Claude calls were made. The chatbot log shows no errors or warnings.
- 2026-11-12: Documents and folders overhaul, backend (the Documents view that uses it comes next). Nested folders (`folders` table, AUTOINCREMENT ids; `documents.folder_id`, NULL is
  Unfiled): names unique per parent ignoring case (casefold), at most 100 characters, no `/`; no folder inside itself
  or its own folders; delete moves the contents up (clashing names become "Name (2)") or, with `contents=delete` and
  the right `confirm_documents` count, deletes the subtree and its documents; uploads on their way in are retargeted
  and chats forget it. Routes: `GET/POST /api/folders`, `PATCH/DELETE /api/folders/<id>`, `POST /api/documents/move`
  (`folder_id` or a `path` that makes missing folders; `from` for Undo), `GET/PATCH /api/documents/<id>` (title,
  type, key details, folder, `suggestion: apply|dismiss`; edited fields kept in `user_edited`, which re-reads and
  Replace keep), `GET /api/documents/<id>/text` and `/thumb`; `?folder=&recursive=&limit=` on `/api/documents` and
  `/api/search`; `upload_types` in `/api/models`. Uploads take `folder_id` and `auto_file`; a duplicate's 409 names
  its folder; Replace keeps the replaced copy's folder; a re-read never moves a document. Analysis returns key details
  (`document_date`, `vendor`, `total_amount`, `currency`, `due_date`; checked in `jobs._details`) and a folder from
  the existing paths: an Unfiled upload with auto-file on goes into an existing folder only on a `high` confidence
  and an exact path, otherwise the suggestion is kept (never a folder made without a click). Photos (JPG, PNG, WebP,
  HEIC with `pillow-heif`, now in requirements.txt): checked at upload (format by the bytes, at most 50 megapixels),
  read by minicpm-v4.6 or by Claude from an image block (Auto local first), estimated for Claude, kept as the type
  of their bytes and opened inline with a sandbox CSP. Markdown (`.md`, `.markdown`) opens as plain text. Thumbnails
  (WebP, 400/1200 px) made in the job and cached in `THUMBS_DIR` by hash, made lazily for older files, served with
  immutable caching when `v` is the hash, removed with their original, not backed up. Chat and streaming take
  `folder_ids`, resolved at question time (subfolders included); answers give `scope_folders` and `scope_count`;
  conversations keep `folder_ids` (added after the AUTOINCREMENT copy). Found on staging: Ollama never shows a
  schema's descriptions to the model, so the date and folder rules sit in the prompt too (`sections.FIELD_RULES`);
  without it qwen3.5:9b wrote "3 November 2026" and filed a receipt as a new "Household/Bills/Receipts" beside the
  existing "Receipts". 243 tests pass (27 new; the `.png` row of the unreadable-files test now expects the photo
  error). Staging (fresh library, real models): a generated receipt photo read by minicpm-v4.6 (text exact) and
  auto-filed into Receipts with date 2026-11-03, Tesco Stores, 42.10 GBP; a water bill PDF auto-filed into
  Household/Bills with bill and due dates, vendor and 84.20 GBP; a manual page uploaded into Tractor stayed there
  with no vendor or total; a `.markdown` file served as text/plain and given back as written; thumbnails 400 px wide;
  a question about Tractor answered from it alone ("38.0 litres", cited with its page); a re-read kept the folder and
  the edited vendor and due date; one receipt read by Claude Haiku 5.5 through an image block (2 calls, $0.0005).
  The current page still works with it all (light and dark, 1400 and 390 px, no console errors). Not done here: the
  page's CSP on `/` (it needs the page split into files first, the frontend's first step)
- 2026-11-12: Documents and folders overhaul: nested folders, a Documents view (grid/list, detail, selection, drag and
  drop), ask about a folder, uploads into folders, photo uploads read by the vision model, auto-filing with
  suggestions, key details, thumbnails, and markdown upload with a sanitized preview. Split index.html into app.css,
  app.js, docs.js and md.js; removed DataTables and jQuery. The page: a Chat | Documents switch (the folder button on
  phones), both views kept alive (an answer keeps streaming, with a dot on Chat), places in the URL hash (reload, Back
  and Forward; on a phone every level is a step back), the left column 288 px with the Add box (uploads go where
  Documents is; auto-file box for Unfiled), the queue and a folder tree (counts, Ask, menus, arrow keys, F2, Delete,
  A, Shift+N), Chats as a drawer in Documents. Documents: All documents, Library, Unfiled and folders, breadcrumbs,
  cards (thumbnails or typed icons, key details, due dates, unit fixes, suggestion chips, "Not analyzed") or a list
  whose columns follow the width, natural sort with empty values last, search here or everywhere (instant, then the
  server's matches and passages), filters as chips, totals per currency, selection (box, Ctrl/Shift-click, Ctrl+A,
  Select on phones) with Ask, Move, Re-read, Delete (two clicks) and New folder from these, a move picker, drag and
  drop of documents and folders and of files onto folders, Undo after every move, ghost cards while uploads are
  read, re-read strips, the suggestions strip and first-run tip. Detail (column, overlay or page): previews (PDF page
  1, photo with its read text, Markdown formatted or plain, text), title and key details edited in place, folder
  line, suggestion banner with Dismiss, summary, keywords, copies, actions. Folder dialogs (new, rename, delete with
  move-up default or a ticked delete-everything checked by the server's count). Scope: chip with folders, a tree
  dialog, "Only in Receipts (folder)", the folder empty state, Show in Documents from a source. `md.js` builds DOM
  only (no innerHTML), links http(s)/mailto only, images as chips, 6 levels and 200 KB at most. `GET /` sends a CSP
  (own scripts only, no inline). 244 tests pass (1 new for the page's files, CSP and static caching; the login test
  now uses `/static/app.js`). Staging (fresh library, real models), Playwright: a layout sweep at every width from
  320 to 1600 px (4 px steps light, 40 px dark), both views, with and without Log out: no sideways scroll, header and
  Documents' parts fit (short header labels now at 821-850 px, 821-880 with a login). First run (no folders): empty
  state, ghost cards, the tip, tree and listing keys, and 3 Unfiled documents filed into a new "Manuals" in 5 taps on
  a phone: 14 checks pass. Seed: a 3-page manual into Tractor/Manuals, a receipt photo (minicpm-v4.6) auto-filed
  into Receipts with its key details, a water bill auto-filed into Household/Bills, Markdown and text files. Flows:
  94 checks pass at 1440/1024/1600 (light and dark) and 390/320 (touch, dark): folders, include subfolders, PDF
  preview "Page 1 of 3", title edit, list sort, Ctrl/Shift-click and Ctrl+A, search with passages and Search
  everywhere, suggestion chip and Undo, picker and Undo, name clash inline, drag and drop of a card and a folder (a
  folder into its own subfolder refused), F2 rename, delete dialog both modes, Markdown preview (a 140 KB hostile file
  renders in 12 ms with no img, no script, no non-http link, 6 nested quotes at most, no request to example.com), a
  receipt's vendor edited and kept by a re-read, filters and sort by amount, reload and a deleted folder's hash, Ask
  about Tractor answered from the manual, then a PDF dropped onto the Manuals tile (ghost card, real card) was cited
  by the next question without changing the scope, switching to Documents while an answer streamed (dot, answer
  finished), the scope dialog, Show in Documents, the medium overlay, and on a phone Back from detail to folder to
  Library to Chat, select mode with the bottom bar, the picker and menu as sheets, the Add sheet with Take a photo,
  and the detail's five-button bar fitting at 320 px and hiding while a field has focus. Found and fixed on the way:
  the list's sticky header sat over the first row (a scrolling wrapper), the Markdown nesting cap allowed 7 levels,
  the phone-only Uploads box and scope toggle showed on desktop, the Add box overflowed with the model picker
- 2026-11-12: Review fixes for the documents and folders overhaul. An upload being read when its folder was deleted
  landed in Unfiled: the worker saved with the options it had claimed the job with. `add_document` now reads the
  job's folder and `auto_file` from its row inside its transaction, and Auto's switch to Claude writes only its own
  keys (`db.set_job_options`, `json_set`) instead of the whole stale dict. Uploads redirected by a folder delete get
  `auto_file` false, so one sent to Unfiled is no longer auto-filed elsewhere. Auto-filing also needs the folder to be
  named for the document (`jobs._related`: a word of the folder's name matches its type, title, vendor or keywords,
  plurals, endings and accents aside), and the prompt says so; `qwen3.5:9b` had filed a home insurance policy into
  Bills and barn notes into Tractor/Manuals. An auto-filed upload's row stays until hidden. Amounts: "1,234.56",
  "12,000,000", "GBP 42.10" and "42.10 GBP" are read (a code or £/€ gives a missing currency); the detail's Total
  takes the same forms and asks about "1,500" instead of saving 1.5. Phone JPEGs that Pillow opens as MPO (iPhone gain
  maps, Android and camera JPEGs) are accepted as JPEGs. Folders go 20 levels deep at most (one request had made
  3,000 and a 33 MB folder listing). A UTF-8 BOM is dropped when indexing and previewing (`utf-8-sig`, and in
  `md.js`). The page: the ⋯/right-click menu of one of several selected documents acts on them all; Undo of a move
  removes the folders it made if still empty; deleting a folder in move mode no longer leaves a surviving subfolder
  being viewed; "Everywhere" resets in each place and the box says "Search everywhere…"; the count follows Include
  subfolders; a finished re-read redraws the open preview; an amount read without a currency counts as the others'
  when they share one (the tooltip says so); the filter gains Read by and Not fully analyzed; the empty-folder check
  is left to the server (the page's listing could be older); a saved conversation whose folder was deleted says
  "Chosen folder deleted". README's stale "Not done yet" line replaced. 251 tests pass (7 new, 1 changed for the name
  check). Staging, real models: 17 API checks (the redirect mid-read, auto-filing of 5 files, an MPO receipt read by
  minicpm-v4.6, a BOM file, depth, empty folder) and 27 Playwright checks pass (1440 light, 1024 and 390/320 dark),
  screenshots looked at; the only console entry is the empty folder's intended 400. No finding was rejected; the old
  table's Read by and Analysis columns came back as filters rather than columns

## To do

- Deploy the documents and folders overhaul (the backend commit a964259 and the Documents view after it) to the live
  app. Neither is deployed yet. The live library's 14 documents will show in Library as Unfiled with typed icons
  (they have no kept originals, so no thumbnails); the first-run tip and "New folder from these" file them.
- Install the port proxy task once, from an elevated PowerShell (README, "Reach it from the LAN").
- Decide the backup time zone: nightly backups run at 03:00 UTC (10 or 11 p.m. Eastern) unless `TZ` is set in `.env`
  (e.g. `TZ=America/New_York`, then `docker compose up -d backup`).
- Decide on a password: the live app has no login until `APP_PASSWORD` is set in `.env` (README, Login), and
  `docker compose up -d chatbot`.
- Try a real scanned manual through the live app. Scanned test PDFs have been read end to end on staging, by
  minicpm-v4.6 and by Claude Haiku (see the log). The 14 live documents have no kept original, so they can't be
  re-read or opened, and they have no page numbers: uploading a manual again (and deleting its old copy) gives it both.
- Once the new version has proven itself, delete the pre-deploy copies: `/data/metabase-before-enhancements-20261112.db`
  in the volume (97 MB) and `~/chatbot-predeploy/` in WSL.
- Before pushing: ae11627, 42a0a88, c31bf23, 3e9eb52 and 7a86b66 end with a `Co-Authored-By: Claude` trailer. Removing
  it means rewording those commits (a history rewrite of 2ca20b9..master), which needs the owner's go-ahead.
- Push, when wanted.
- Done: the enhancement plan is deployed (see the log). The Windows NVIDIA driver is 582.78 and Ollama runs on the GPU
  (GTX 1080 Ti). The UI has been checked in a browser, light and dark, at phone widths (Playwright, see the log).
  Everything is committed.

## Left on the machine

- Containers `chatbot-chatbot-1` (the enhancement plan, deployed 2026-11-12), `chatbot-backup-1` and `chatbot-ollama-1`
  are running inside WSL `vostok`, with `restart: unless-stopped`; the library is in volume `chatbot_metabase`, the
  nightly snapshots in `chatbot_backups`.
- A way back: the old code is tagged `chatbot-chatbot:894b984` and runs on the migrated library. In the repo folder in
  WSL: `docker compose stop chatbot backup`, then `docker run -d --name chatbot-rollback --restart unless-stopped
  --network chatbot_default -p 0.0.0.0:8000:8000 -v chatbot_metabase:/data --env-file .env chatbot-chatbot:894b984`.
  Copies of the library from just before the deploy are in the volume (`/data/metabase-before-enhancements-20261112.db`)
  and in WSL (`~/chatbot-predeploy/`).
- WSL is kept running by a `wsl -d vostok -- sleep infinity` started from a Claude Code session on 2026-11-11.
  After a reboot, nothing starts WSL until the port proxy task (plan item 13) is installed.
- Open to the LAN through the port proxy `0.0.0.0:8000` to WSL's address and the firewall rule "Chatbot 8000".
  From an elevated prompt, remove with:
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

- [x] 12. A login. The app is open to the LAN on port 8000 with no password; anyone can delete
      documents or change the instructions. A single shared password would close that.
- [x] 13. Stable network address. The Windows portproxy points at the WSL IP, which can change on
      restart. Use WSL mirrored networking or a start-up script that re-points it.
      (Start-up script; its scheduled task still has to be installed once, elevated.)
- [x] 14. Backups. The library is one SQLite file in volume `chatbot_metabase`. Nightly copy.
- [x] 15. Run the tests on GitHub (Actions workflow on every push); today they run by hand.
- [x] 16. Usage log. Record tokens and model per request to see what Claude costs.
