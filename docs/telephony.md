# Telephony & SIP Outbound Configuration

This document describes how to configure LiveKit SIP outbound trunking for real outbound phone calls (Milestone M2).

Trunk setup is performed manually using the LiveKit CLI (`lk`). **Never commit real credentials or `outbound-trunk.json` to the repository** (it is added to `.gitignore`).

---

## 1. Prerequisites

1. Install the LiveKit CLI:
   ```bash
   # macOS / Linux
   curl -sSL https://get.livekit.io/cli | bash
   ```
2. Authenticate the CLI with your LiveKit Cloud or self-hosted project:
   ```bash
   lk project authenticate
   ```
   Or set environment variables:
   ```bash
   export LIVEKIT_URL="wss://your-project.livekit.cloud"
   export LIVEKIT_API_KEY="your-api-key"
   export LIVEKIT_API_SECRET="your-api-secret"
   ```

---

## 2. Prepare `outbound-trunk.json`

Copy the provided example template:
```bash
cp outbound-trunk.example.json outbound-trunk.json
```

Edit `outbound-trunk.json` with credentials from your SIP provider (Telnyx or Twilio).

### Telnyx Example
In Telnyx, create a SIP Trunk under **Voice > SIP Trunks**, assign your purchased phone number, and set up SIP Authentication credentials (FQDN: `sip.telnyx.com`):

```json
{
  "name": "telnyx-outbound",
  "address": "sip.telnyx.com",
  "transport": "SIP_TRANSPORT_AUTO",
  "numbers": [
    "+48123456789"
  ],
  "auth_username": "your_telnyx_sip_username",
  "auth_password": "your_telnyx_sip_password"
}
```

### Twilio Example
In Twilio, create an Elastic SIP Trunk under **Voice > Manage > Trunks**, set Termination SIP URI (e.g., `your-trunk.pstn.twilio.com`), and configure authentication credentials:

```json
{
  "name": "twilio-outbound",
  "address": "your-trunk.pstn.twilio.com",
  "transport": "SIP_TRANSPORT_AUTO",
  "numbers": [
    "+48123456789"
  ],
  "auth_username": "your_twilio_sip_username",
  "auth_password": "your_twilio_sip_password"
}
```

---

## 3. Create the Outbound Trunk in LiveKit

Run the LiveKit CLI command:
```bash
lk sip outbound create outbound-trunk.json
```

The output will display the created trunk details, including the Trunk ID starting with `ST_`:
```text
Trunk ID: ST_xxxxxxxxxxxx
Name: telnyx-outbound
Address: sip.telnyx.com
...
```

Copy the Trunk ID (`ST_xxxxxxxxxxxx`).

---

## 4. Configure `.env`

Add the trunk ID and an explicit agent dispatch name to your local `.env`:

```bash
SIP_OUTBOUND_TRUNK_ID=ST_xxxxxxxxxxxx
AGENT_DISPATCH_NAME=callback-agent
```

> [!IMPORTANT]
> **Browser Mode vs Phone Mode with `AGENT_DISPATCH_NAME`:**
> When `AGENT_DISPATCH_NAME` is set, the worker registers with that explicit dispatch name and **will no longer join LiveKit Playground rooms automatically**.
> Outbound phone calls require explicit dispatch via `call_me.py`.
> To return to testing in the browser via LiveKit Agents Playground, leave `AGENT_DISPATCH_NAME` empty (`AGENT_DISPATCH_NAME=`).

---

## 5. Running a Test Call

1. In one terminal, start the voice agent worker in dev mode:
   ```bash
   uv run python -m agent.worker dev
   ```
   *(Ensure `AGENT_DISPATCH_NAME=callback-agent` and `SIP_OUTBOUND_TRUNK_ID=ST_...` are set in `.env`)*

2. In another terminal, trigger the outbound phone call:
   ```bash
   uv run python scripts/call_me.py +48XXXXXXXXX
   ```

3. What happens:
   - `call_me.py` validates the number as E.164, creates a room, and dispatches the agent.
   - The worker enters phone mode, starts audio pipelines, and calls `create_sip_participant` with `wait_until_answered=True`.
   - Your phone rings.
   - When you answer, the agent greets you and conducts the qualification conversation from `agent/script.yaml`.
   - When finished (or upon saying goodbye), the `end_call` tool speaks the closing phrase to completion, waits for playout, and hangs up by deleting the room.
