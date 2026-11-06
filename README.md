# Veyra

A Python pipeline for analysing a reference video and recreating it. The working code lives in the `refclone` package.

## Status

Early-stage. The project has a planned architecture and a working FastAPI service skeleton. The video analysis and generation stages are still to be built.

## What exists today

- A FastAPI service in `refclone/src/refclone` with a `/health` endpoint
- An API test in `refclone/tests`
- Dependencies chosen for audio and voice work: pydub, numpy, torch and coqui-tts
- Planning documents covering the design and implementation steps: `planforreferencevideo.md`, `planforreferencevideo-implementation.md` and `refclone/docs`

## Run it locally

```
cd refclone
pip install -r requirements.txt
uvicorn refclone.main:app --app-dir src --reload
pytest
```

Then open http://127.0.0.1:8000/health

## Stack

Python, FastAPI, Uvicorn, pytest, pydub, numpy, PyTorch, Coqui TTS.
