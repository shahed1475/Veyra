from fastapi import FastAPI

app = FastAPI(title="refclone")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}
