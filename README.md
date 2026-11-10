# Document Chatbot (PoC)

A small standalone chatbot that answers questions with the Claude API, using your own API key. You upload documents, Claude pulls metadata out of each one (title, type, summary, keywords), and everything goes into an embedded SQLite database with full-text search. When you ask a question, the app finds the most relevant passages and sends them along with it.

## Run it

```bash
docker compose up --build
```

Open http://localhost:8000, paste your Claude API key into the sidebar, and upload some files.

The key stays in your browser's localStorage and goes along with each request. The server never stores it. If you'd rather set it on the server, export `ANTHROPIC_API_KEY` before running `docker compose up`.

The database is a single file, `/data/metabase.db`, kept in the `metabase` Docker volume. No database server is needed.

## What it does

| Feature | How |
|---|---|
| Upload | PDF (text-based), DOCX, and any UTF-8 text file (txt, md, csv, json, code…) |
| Analyze | One Claude call per document with structured JSON output: title, doc type, summary, keywords |
| Metadata base | SQLite tables `documents` and `chunks`, plus FTS5 indexes over the metadata and the full text (porter stemming) |
| Search | The sidebar search box matches documents by metadata and lists matching passages with highlighted snippets |
| Chat | Each question runs a full-text search for the top 5 passages, which go into the prompt. Answers cite file names |

## Keeping token use down

- **Retrieval instead of whole documents:** a chat turn sends only the top-K passages (`TOP_K`, default 5, about 1,500 characters each), not entire files.
- **Prompt caching:** chat requests set `cache_control`, so the system prompt and the earlier conversation are read from cache (about 10% of the normal input price). The browser sends back the exact earlier turns so the prefix stays byte-identical. Each answer shows how many tokens were read from and written to the cache.
- **Bounded analysis:** metadata extraction only reads the first `ANALYZE_CHARS` (default 40,000) characters of a document. The full text is still indexed and searchable. Documents analyzed this way are marked `partial`.
- **Low effort by default:** `CLAUDE_EFFORT=low`. Raise it if you want more careful answers.

## Configuration

| Env var | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | Optional server-side key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | For example `claude-haiku-5-5` to cut costs |
| `CLAUDE_EFFORT` | `low` | `low` / `medium` / `high` / `xhigh` / `max` |
| `SYSTEM_PROMPT` | built-in | Gives the bot its personality |
| `ANALYZE_CHARS` | `40000` | Characters per document sent for metadata extraction |
| `TOP_K` | `5` | Passages retrieved for each question |
| `MAX_UPLOAD_MB` | `25` | Maximum upload size |
| `DB_PATH` | `/data/metabase.db` | Location of the SQLite file |

Requests turn on server-side refusal fallbacks (`fallbacks: "default"`). If Claude's safety filter declines a request, the API retries it on a fallback model instead of failing.

## Develop without Docker

```bash
pip install -r requirements.txt pytest
DB_PATH=data/metabase.db flask --app app.main run --debug
pytest
```

## Not done yet

- Scanned PDFs (would need OCR)
- Batch analysis for bulk uploads (the Message Batches API is 50% cheaper but asynchronous)
- Semantic or embedding search (FTS5 matches keywords, not meaning)
- Authentication (anyone who can reach the port can see the documents)
