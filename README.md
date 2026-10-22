# Voice Callback Agent

An AI voice callback system designed for rapid lead response. When a prospective lead submits their name and phone number on a website form, the system automatically initiates an outbound call within 30 seconds. The conversational AI agent introduces itself, asks a short set of qualifying questions, and after the call completes, generates a structured summary delivered to a sales manager (via Telegram) and pushed to a CRM.

## Stack

- **Language & Runtime:** Python 3.12 (managed via `uv`)
- **Web Framework:** FastAPI, Uvicorn
- **Database:** PostgreSQL 16 (asyncpg, SQLAlchemy 2.0, Alembic)
- **Voice / Telephony:** LiveKit Agents, LiveKit SIP (Twilio / Telnyx)
- **Containerization:** Docker & Docker Compose
- **Quality & Testing:** Pytest, Ruff

## Project Structure

```
app/
  main.py                   # FastAPI entrypoint & healthcheck
  config.py                 # Pydantic Settings (loads from .env)
  db.py, models.py          # SQLAlchemy models (Lead, Call, CallResult)
  api/
    leads.py                # Public: POST /leads & demo form
    internal.py             # Agent callbacks (shared-secret auth)
  services/
    dispatch.py             # LiveKit agent dispatching
    summarizer.py           # Call transcript summarization
  integrations/
    crm/base.py             # CRMAdapter interface
    crm/sheets.py           # Demo adapter (Google Sheets)
    notify/telegram.py      # Telegram manager notification
agent/
  worker.py                 # LiveKit agent worker entrypoint
  prompts.py                # System prompt generator
  tools.py                  # Agent tools (save_answer, end_call, voicemail)
  script.yaml               # Configurable conversation script
tests/
  test_health.py            # API health endpoint tests
docker-compose.yml          # PostgreSQL 16 local database
.env.example                # Example environment variables
```

## Getting Started

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose

### 1. Configure Environment

Copy the example environment configuration:

```bash
cp .env.example .env
```

All credentials are fully optional for local development startup.

### 2. Start PostgreSQL

Launch the PostgreSQL 16 container:

```bash
docker compose up -d
```

To stop the database:

```bash
docker compose down
```

### 3. Install Dependencies

Install the runtime and development dependencies using `uv`:

```bash
uv sync
```

### 4. Run the API Server

Start the FastAPI application with `uvicorn`:

```bash
uv run uvicorn app.main:app --reload
```

The server will be available at `http://localhost:8000`. You can verify the health endpoint:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

API documentation is accessible at `http://localhost:8000/docs`.

### 5. Run the Voice Agent Worker (M1 Browser Agent)

The voice agent connects to LiveKit Cloud (or self-hosted LiveKit) and lets you converse with it in real time directly from your browser.

#### Required Environment Variables
In `.env`, configure your credentials:

```bash
# LiveKit Cloud credentials
LIVEKIT_URL=wss://<project>.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# AI Providers (Free tiers)
LLM_PROVIDER=google
LLM_MODEL=gemini-3.5-flash-lite
# Optional fallback model (e.g. gemini-3.5-flash)
# If set, primary and fallback models are wrapped with LiveKit's LLM FallbackAdapter
LLM_FALLBACK_MODEL=gemini-3.5-flash
GOOGLE_API_KEY=your_google_ai_studio_api_key

STT_PROVIDER=deepgram
# Language is configured in agent/script.yaml (default: uk); set STT_LANGUAGE only to override:
# STT_LANGUAGE=
DEEPGRAM_API_KEY=your_deepgram_api_key

TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Dispatch: leave empty for automatic dispatch (LiveKit Playground / dev)
# Set to a name (e.g. callback-agent) for explicit dispatch (outbound calls via API)
AGENT_DISPATCH_NAME=

# Turn detection: "vad" (local Silero, free) or "turn_detector" (inference)
TURN_DETECTION_MODE=vad
# Interruption mode: "vad" (local Silero, free) or "adaptive" (inference)
INTERRUPTION_MODE=vad
```

> **Note on LLM Fallback & Error Handling:**
> - When `LLM_FALLBACK_MODEL` is configured, LiveKit's `FallbackAdapter` seamlessly routes generation to the fallback model if the primary model errors out (e.g., transient HTTP 503 high demand or rate limits).
> - If all retries and fallbacks fail, the agent will not stay silent: it speaks a fixed apology phrase from `agent/script.yaml` (`error_phrase`, e.g. in Ukrainian: *"Перепрошую, у нас виникла технічна помилка..."*) with interruptions disallowed, waits for playout, and gracefully terminates the session via `aclose()`.

> **Note on Language:** The single source of truth for the call language is `agent/script.yaml` (default: `"uk"`). Switching to Polish or English is done by changing `language` and the phrases in `agent/script.yaml`.

> **Note on Dispatching:** When `AGENT_DISPATCH_NAME` is left empty, the worker registers as a default room agent so it automatically joins any LiveKit Agents Playground room. If set to a specific name, LiveKit will only dispatch the agent when explicitly requested by name.

> **Note on Swappability:** To switch to OpenAI without code changes, set `LLM_PROVIDER=openai`, `STT_PROVIDER=openai`, `TTS_PROVIDER=openai`, and provide `OPENAI_API_KEY`.

#### Start the Worker in Development Mode
Run the agent worker:

```bash
uv run python -m agent.worker dev
```

*(Alternatively, if you have the LiveKit CLI installed, you can use `lk agent dev` for hot reloading).*

The worker registers itself with LiveKit Cloud and waits for incoming room connections.

#### Connecting via LiveKit Agents Playground
1. Open the [LiveKit Agents Playground](https://agents-playground.livekit.io/) in your browser.
2. Connect using your LiveKit Cloud project credentials (or generate a connection token in your LiveKit Cloud dashboard).
3. Allow microphone permissions.
4. When you connect, the agent introduces itself with the greeting from `agent/script.yaml`:
   > *"Доброго дня! Мене звати Анна, я віртуальна асистентка компанії Solaris Tech. Попереджаю, що розмова може записуватися. Телефоную у відповідь на заявку, яку ви залишили на нашому сайті. Чи підкажете, як я можу до вас звертатися?"*
5. Answer the qualifying questions (name, service, timeline, preferred callback time).
6. Interruptions are enabled: speak while the agent is talking to test interruption handling and turn detection.
7. When the call ends, inspect the worker console log for the structured JSON session summary containing the full transcript and collected answers.

### 6. Run Tests and Linting

Run test suite:

```bash
uv run pytest
```

Run linter and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```
