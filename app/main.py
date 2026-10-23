from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.calls import router as calls_router
from app.api.events import router as events_router
from app.api.internal import router as internal_router
from app.api.leads import router as leads_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Voice Callback Agent",
    description="Intake, call orchestration, and post-call processing backend.",
    version="0.1.0",
)

app.include_router(internal_router)
app.include_router(leads_router)
app.include_router(calls_router)
app.include_router(events_router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
async def overview_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/calls", response_class=FileResponse)
async def calls_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "calls.html", media_type="text/html")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Strips echoed input from /leads 422s so a raw phone never reaches a response."""
    if request.url.path != "/leads":
        return await request_validation_exception_handler(request, exc)
    errors = [
        {k: v for k, v in err.items() if k in ("type", "loc", "msg")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
