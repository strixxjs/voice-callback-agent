# Demo plan

Goal: the client leaves a number on a demo form → phone rings within 30 s → natural conversation → summary appears in Telegram + a sheet (CRM stand-in).

Rough total: 20–25 h for minimal demo, 30–50 h with polish.

## M0 — Setup (1–2 h)
- [ ] Private GitHub repo, `uv` project, ruff, pytest, `.env.example`
- [ ] Docker Compose with Postgres
- [ ] Accounts: LiveKit Cloud, OpenAI, Deepgram, ElevenLabs/Cartesia, Twilio or Telnyx

## M1 — Voice agent in the browser, no phone (4–6 h)
- [ ] Minimal LiveKit agent worker, talk to it via LiveKit Agents Playground
- [ ] System prompt from `agent/script.yaml`
- [ ] Interruptions and turn detection work
- **Done when:** a 2-minute conversation feels natural

## M2 — Real outbound phone call (4–8 h, highest risk)
- [ ] SIP trunk (Twilio/Telnyx) connected to LiveKit outbound
- [ ] Script `scripts/call_me.py <phone>` dispatches the agent and my phone rings
- [ ] `end_call` tool hangs up correctly
- **Done when:** the agent calls my real phone and ends the call itself

## M3 — Conversation quality (6–10 h)
- [ ] `save_answer` tool, all required fields collected
- [ ] Pick STT/TTS by quality in the target language
- [ ] Voicemail / no-answer handling
- [ ] Measure latency (target < 1 s response), tune
- [ ] 10+ test calls, fix the prompt after each

## M4 — Backend flow (4–6 h)
- [ ] `POST /leads` + simple demo HTML form with consent checkbox
- [ ] Lead saved → agent dispatched automatically (< 30 s)
- [ ] Agent posts transcript to `/internal/calls/{id}/completed`
- [ ] Working-hours and retry rules

## M5 — Summary and delivery (3–5 h)
- [ ] Summarizer → `CallSummary` (Pydantic): fields, summary, lead score, next step
- [ ] Telegram notification to manager
- [ ] `CRMAdapter` + Google Sheets adapter

## M6 — Demo readiness (3–5 h)
- [ ] Deploy (backend + worker)
- [ ] Rehearse the live demo end to end 3 times
- [ ] Short cost estimate per call for the client

## After the client says "yes"
- Real CRM adapter, admin panel, call recordings, analytics, load testing, multiple languages.
