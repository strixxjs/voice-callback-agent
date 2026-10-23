from fastapi import FastAPI

from app.api.internal import router as internal_router

app = FastAPI(
    title="Voice Callback Agent",
    description="Intake, call orchestration, and post-call processing backend.",
    version="0.1.0",
)

app.include_router(internal_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
