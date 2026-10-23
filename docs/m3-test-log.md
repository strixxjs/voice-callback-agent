# M3 Conversation Quality — Test Log

This log tracks test conversations carried out in LiveKit Agents Playground / browser mode to measure response latency, voice quality, and conversation flow across iterations.

## Test Conversations Log

| # | Date | STT | TTS Model + Voice | LLM | Avg / Max Latency | What went wrong | Prompt change made |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *e.g. 1* | *2026-10-23* | *Deepgram nova-3 (uk)* | *ElevenLabs eleven_turbo_v2_5 (voice: hpp4J3VqNfWAUOO0d1Us)* | *Gemini 3.5 Flash Lite* | *1.15s / 1.48s* | *Interrupted too early during a mid-sentence pause.* | *Increased endpointing min delay to 0.7s.* |
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |

## How to extract metrics for the table
Each test call logs a session summary at completion:
```
Session metrics summary: room=call-xyz, turns=4, avg=1.120s, p50=1.050s, max=1.350s | LLM=google/gemini-3.5-flash-lite, STT=deepgram/nova-3, TTS=elevenlabs/eleven_turbo_v2_5 (voice=hpp4J3VqNfWAUOO0d1Us)
```
Per-turn JSON lines are stored under `logs/metrics/<room>.jsonl`.
