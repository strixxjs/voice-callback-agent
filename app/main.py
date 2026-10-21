from fastapi import FastAPI

app = FastAPI(
    title="Voice Callback Agent",
    description="Intake, call orchestration, and post-call processing backend.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
