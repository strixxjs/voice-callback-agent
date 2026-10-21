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

### 5. Run Tests and Linting

Run test suite:

```bash
uv run pytest
```

Run linter and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```
