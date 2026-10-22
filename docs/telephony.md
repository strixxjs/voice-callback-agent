# Telephony & SIP Outbound Configuration

This document describes how to configure LiveKit SIP outbound trunking for real outbound phone calls (Milestone M2).

Trunk setup is performed using the LiveKit CLI (`lk`). **Never commit real credentials or `outbound-trunk.json` to the repository** (it is ignored via `.gitignore`).

---

## 1. Prerequisites

1. Install the LiveKit CLI:
   ```bash
   # macOS / Linux
   curl -sSL https://get.livekit.io/cli | bash
   ```
2. Export your LiveKit credentials from `.env` into the current shell session:
   ```bash
   set -a; source .env; set +a
   ```
   The `lk` CLI automatically reads `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` from environment variables.

---

## 2. Prepare `outbound-trunk.json`

Copy the provided example template:
```bash
cp outbound-trunk.example.json outbound-trunk.json
```

The CLI expects configuration wrapped in a top-level `"trunk"` object.

> [!NOTE]
> **Caller ID (`numbers`):**
> The `"numbers"` array specifies your purchased provider phone number(s) used as caller ID (`From`), **not** the destination phone number you are dialing. Always provide this in E.164 format (e.g. `+1XXXXXXXXXX`).

### Telnyx Configuration

Refer to the official [LiveKit Telnyx Integration Guide](https://docs.livekit.io/telephony/start/providers/telnyx/) for full provider setup details.

Key requirements in Telnyx:
- Create an **FQDN Connection** with credentials authentication. The FQDN configured in Telnyx points to your LiveKit SIP endpoint (e.g. `xxxx.sip.livekit.cloud`), while `"sip.telnyx.com"` is the SIP destination address defined in your LiveKit trunk configuration.
- Assign an **Outbound Voice Profile** to the connection and ensure your destination country's dialing permissions are enabled.
- Assign your purchased phone number to the FQDN connection.

```json
{
  "trunk": {
    "name": "telnyx-outbound",
    "address": "sip.telnyx.com",
    "numbers": [
      "+1XXXXXXXXXX"
    ],
    "auth_username": "your_telnyx_sip_username",
    "auth_password": "your_telnyx_sip_password"
  }
}
```

### Twilio Configuration

In Twilio, create an Elastic SIP Trunk under **Voice > Manage > Trunks**, set the Termination SIP URI (e.g., `your-trunk.pstn.twilio.com`), and configure authentication credentials under **Credentials Lists**:

```json
{
  "trunk": {
    "name": "twilio-outbound",
    "address": "your-trunk.pstn.twilio.com",
    "numbers": [
      "+1XXXXXXXXXX"
    ],
    "auth_username": "your_twilio_sip_username",
    "auth_password": "your_twilio_sip_password"
  }
}
```

### Provider Trial Limitations

- **Telnyx Trial:** Accounts on a trial plan can only make outbound calls to pre-verified phone numbers.
- **Twilio Trial:** Elastic SIP Trunking requires an upgraded account; trial accounts do not support outbound SIP trunks.

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
