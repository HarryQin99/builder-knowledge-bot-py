from fastapi import FastAPI

app = FastAPI(title="Knowledge Bot")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
