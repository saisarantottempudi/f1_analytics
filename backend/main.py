"""FastAPI entry point — full endpoints land in stage 5."""

from fastapi import FastAPI

app = FastAPI(title="F1 Analytics AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "stage": "1-scaffold"}
