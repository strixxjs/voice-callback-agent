# Voice Callback Agent — project context

## What we are building
A demo (that must be able to grow into production) for a client:

1. A person leaves their name + phone on the client's website form.
2. Within **30 seconds** an AI voice agent calls them back by phone.
3. The agent has a **natural, well-tuned conversation**: introduces itself as an AI assistant, asks a short set of qualifying questions, handles interruptions, says goodbye and hangs up.
4. After the call, a second LLM step turns the transcript into a **short summary + structured fields**.
5. The result is delivered to a **real person** (sales manager) and pushed to a **CRM**.

The CRM is **not known yet**. Never hard-code a specific CRM — everything goes through the `CRMAdapter` interface. For the demo we use a simple adapter (Google Sheets or console/DB).

Quality of the conversation is the #1 priority. A working but robotic call = failed demo.

## Stack
- Python 3.12, `uv` for dependencies
- **FastAPI** — lead intake, call orchestration, post-call processing, internal API
- **PostgreSQL** + SQLAlchemy 2.0 (async) + Alembic
- **LiveKit Agents** (Python) — the voice agent worker, deployed to LiveKit Cloud or self-hosted
- **LiveKit SIP** + SIP trunk (Twilio or Telnyx) — outbound phone calls
- STT / LLM / TTS providers are **configurable via env** (initial choice: Deepgram STT, OpenAI LLM, ElevenLabs or Cartesia TTS). Choose by quality in the call language, not by habit.
- Telegram bot — notification to the manager
- Docker Compose for local dev, `pytest`, `ruff`

LiveKit's API changes often. **Before writing LiveKit code, check the current official docs and examples** (agents, outbound calls, SIP) instead of relying on memory.

## Architecture
```
Website form ──POST /leads──▶ FastAPI ──▶ DB (lead, status=new)
                                 │
                                 └─ dispatch agent via LiveKit API
                                    (metadata: lead_id, name, phone)
                                         │
                             LiveKit agent worker
                             - creates SIP participant → phone rings
                             - conversation (STT → LLM → TTS)
                             - tools: save_answer, end_call
                                         │
                       POST /internal/calls/{id}/completed (transcript)
                                         │
                FastAPI: summarizer (structured output, Pydantic)
                         → DB → CRMAdapter → Telegram notification
```

## Folder structure (target)
```
app/
  main.py
  config.py              # pydantic-settings, all secrets from env
  db.py, models.py       # Lead, Call, CallResult
  api/leads.py           # public: POST /leads (+ demo HTML form)
  api/internal.py        # agent → backend callbacks (shared-secret auth)
  services/dispatch.py   # starts the call via LiveKit
  services/summarizer.py # transcript → CallSummary (Pydantic)
  integrations/crm/base.py      # CRMAdapter protocol
  integrations/crm/sheets.py    # demo adapter
  integrations/notify/telegram.py
agent/
  worker.py              # LiveKit agent entrypoint
  prompts.py             # system prompt built from script.yaml
  tools.py               # save_answer, end_call, detect voicemail
  script.yaml            # questions, language, company info — editable without code
tests/
docker-compose.yml
.env.example
```

## Business rules (do not break)
- Call only if the lead gave **consent** (checkbox on the form).
- Agent **says it is an AI assistant** at the start and that the call may be recorded.
- No calls outside allowed hours (configurable, default 9:00–20:00, client timezone). Outside hours → queue for next morning.
- No answer / voicemail → do not leave a long message; mark status and retry once after N minutes (configurable).
- Agent must not invent prices, promises or facts not present in `script.yaml`. If unsure → "a manager will get back to you".
- Calls short: target 2–4 minutes.
- Phone numbers normalized to E.164.

## Open questions (client has not answered yet — keep these configurable)
- Call language (Polish / Ukrainian / English?)
- Exact questions to ask and fields to collect
- Which CRM
- Expected lead volume per day
- Retry policy, working hours

## Working rules for Claude Code
- Work **one milestone at a time** from `docs/PLAN.md`. Don't jump ahead.
- Small, reviewable changes. Explain what you changed and how to run/test it.
- Ask before adding a new dependency or external service.
- Type hints everywhere, async I/O, no secrets in code — only `.env`.
- Every external integration behind an interface so it can be swapped.
- Commit messages: `feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`, `test: ...`.
- Log every call state change (lead_id, call_id, status) — debugging calls without logs is impossible.
