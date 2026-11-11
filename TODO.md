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
