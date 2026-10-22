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
- [x] Outbound call code: dispatch + SIP participant via LiveKit (unit-tested, not tested on a real trunk)
- [x] `scripts/call_me.py <phone>` dispatches the agent
- [x] `end_call` hangs up in phone mode (unit-tested)
- [ ] SIP trunk connected and my phone rings — moved to Phone go-live
- Note: Live phone test deferred until the client confirms the demo — see Phone go-live.
- **Done when:** the agent calls my real phone and ends the call itself

## M3 — Conversation quality (6–10 h)
- [x] `save_answer` tool, all required fields collected
- [ ] Pick STT/TTS by quality in the target language
- [ ] Measure latency (target < 1 s response), tune
- [ ] 10+ test calls, fix the prompt after each

## M4 — Backend flow (4–6 h)
- [ ] `POST /leads` + simple demo HTML form with consent checkbox
- [ ] Lead saved → agent dispatched automatically (< 30 s)
- [ ] Test mode: a lead from the demo form dispatches the agent into a room that I join from the browser instead of a phone call, so the whole flow can be tested for $0
- [ ] Agent posts transcript to `/internal/calls/{id}/completed`
- [ ] Working-hours and retry rules

## M5 — Summary and delivery (3–5 h)
- [ ] Summarizer → `CallSummary` (Pydantic): fields, summary, lead score, next step
- [ ] Telegram notification to manager
- [ ] `CRMAdapter` + Google Sheets adapter

## Phone go-live (before the demo, ~1 day)
Note: costs money, start only after the client says yes.
- [ ] Buy a number (US = instant; PL needs documents, days)
- [ ] SIP connection + outbound trunk per `docs/telephony.md`
- [ ] First live call
- [ ] Voicemail / no-answer handling
- [ ] Latency over the phone

## M6 — Demo readiness (3–5 h)
- [ ] Deploy (backend + worker)
- [ ] Rehearse the live demo end to end 3 times
- [ ] Short cost estimate per call for the client

## After the client says "yes"
- Real CRM adapter, admin panel, call recordings, analytics, load testing, multiple languages.
