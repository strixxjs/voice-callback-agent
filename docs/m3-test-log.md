# M3 Conversation Quality — Test Log

Browser mode via the demo console. LLM gemini-3.5-flash-lite (fallback gemini-3.1-flash-lite), STT Deepgram (uk), TTS ElevenLabs.
Latency "meas." = SDK e2e (user stopped speaking → first agent audio); scripted phrases (goodbye, error) are excluded from p50/max from run 3 on.
"Split turns" = consecutive user turns (auto). "Lost words" = checked manually in the transcript.

## Runs — 2026-10-24

| Run | Call | Variant (.env) | Turns | p50 meas. | max meas. | p50 comp. | Split turns | Lost words | Extraction | Notes |
| :-- | :--- | :------------- | :---- | :-------- | :-------- | :-------- | :---------- | :--------- | :--------- | :---- |
| 1 | 3c619a4e | fixed 1.0 + grace 1.2, nova-3, turbo | 4 | 4.704s | 7.671s | 3.935s | 2 | 0 | 3/4 | Gemini slow (LLM TTFT 2–4.7 s), excluded. Grace kept "Сьогодні… о п'ятнадцятій". Optional "timeline" not asked → prompt rule added |
| 2 | 0ce8ca3e | same | 7 | 1.204s | 2.796s | 1.674s | 0 | 0 | 2/4 | Skip scenario: 2 questions declined, agent moved on. Goodbye interrupted with a question → answered, short goodbye. max includes scripted goodbye (before fix) |
| 3 | 616822b8 | same | 4 | 1.295s | 1.705s | 1.819s | 2 | 0 | 4/4 | Grace kept "після обіду". Goodbye interrupted with price question + name correction → agent ignored both → prompt rule strengthened |
| 4 | be525053 | same | 5 | 2.715s | 5.285s | 2.218s | 1 | 0 | 4/4 | Gemini slow on turns 1–2 (TTFT ~2 s). Grace kept "після обіду". Goodbye interrupt OK |
| 5 | 445177c1 | same | 4 | 1.359s | 1.379s | 1.833s | 2 | 0 | 4/4 | Grace kept "у середу" |
| 6 | 855edae7 | same | 4 | 4.444s | 8.230s | 4.729s | 3 | 0 | 4/4 | Gemini slow (TTFT 2.9–4.2 s), excluded. Grace kept "Завтра… О… 15" |
| 7 | 5fe2f024 | same | 5 | 1.613s | 1.953s | 1.795s | 0 | 0 | 4/4 | Clean |
| 8 | bba84215 | same (.env not switched) | 7 | 1.393s | 4.718s | 1.948s | 0 | 0 | 4/4 | New goodbye rule OK: answered price, apologised and used corrected name |
| 9 | 8d622d9f | TTS eleven_flash_v2_5 | 6 | 1.550s | 2.704s | 1.863s | 0 | 0 | 4/4 | Flash ≈ turbo on latency; subjectively slightly better. One clarification asked on unclear answer |
| 10 | 9e3728ce | STT nova-2 + flash | 4 | 1.334s | 2.902s | 1.766s | 0 | 0 | 3/4 | Call time unrecognised ("Я доходь зараз") → missing |

## STT errors (uk)

- nova-3: "Іван" instead of "Євген" (twice), "Я її вагель" (= я Євген), "Сьогодні було загадно", "добачно" / "де Бащня" (= до побачення), "Ейсеп", "Якомога жна скоріше".
- nova-3, 2026-10-23: "свисті стіл", "Звата", "починати свій автомобіль", "зробити свою воду" (= своє видавництво), "Через ти ж же" (= через тиждень), "Малий шеф" (surname), "проблему зі судом".
- nova-2: "свого шиї агента" (= ШІ-агента), "Я доходь зараз" (time lost).

## Conclusions

- Chosen config (new defaults): fixed endpointing min_delay 1.0 s + end_call grace 1.2 s. Preemptive generation hides LLM time inside the endpointing wait.
- Criteria: last answer never lost in all runs (grace caught every continuation); extraction 4/4 whenever the caller answered; p50 1.30–1.61 s when Gemini is in normal state; no interruptions of the caller observed on 1.5 s mid-phrase pauses (turn split, agent did not speak over).
- Goodbye: interruptible; real speech continues the call, noise does not; second goodbye is short.
- STT: keep nova-3 (nova-2 not better; keyterms only on nova-3). Ukrainian STT is the weakest link → for prod: per-call keyterm with the lead's name, evaluate other STT providers.
- TTS: turbo stays the code default; flash v2.5 is equivalent. A more natural voice is a prod task (Multilingual v2/v3, 2–3 voices).
- Gemini Free Tier latency is unstable during the day (TTFT up to 4.7 s) — measure latency only when it is normal.

## Known issues (next)

- When grace catches a continuation, the next turn's measured latency is inflated (anchor on the earlier user segment) — fix together with the console metrics panel.
