# Reference-Video-Driven Video Generator — Phases 0-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest slice of the reference-video-driven generator that ships a real, working end-to-end mode — "voice-only clone" — runnable from the CLI: clone a consenting speaker's voice from a reference video, synthesize a script in that voice, and lip-sync your own footage to it.

**Architecture:** A new local-only Python project (`refclone/`) hosting a FastAPI service with three capabilities (voice enrollment, XTTS-v2 speech synthesis, MuseTalk lip-sync) plus two CLI scripts that chain them together. No UI or job queue yet — those are later phases (4-8) covered by `planforreferencevideo.md`. Every entry point requires an explicit consent confirmation before it will run.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pytest, ffmpeg (system binary), pydub, coqui-tts (XTTS-v2), PyTorch with CUDA, MuseTalk (vendored, invoked as a subprocess).

## Global Constraints

- Local-only: no cloud APIs anywhere in this pipeline (matches `vidgen`'s local-only philosophy — local Ollama + local Kokoro).
- No "paste a URL" reference ingestion — reference videos and clips are local file paths / uploads only.
- Every CLI entry point and API path that runs a cloning pipeline must refuse to execute without an explicit consent confirmation (`--i-have-consent` flag for CLI; this is an interim safeguard until the DB-backed consent gate lands in Phase 5's job queue, per `planforreferencevideo.md` §2).
- Target hardware: local NVIDIA GPU, 8GB+ VRAM.
- New project lives at `refclone/`, a sibling directory to `vidgen/` and `avatar-gen/`, with its own git repo.
- Python package layout: source under `refclone/src/refclone/`, tests under `refclone/tests/`, one-off runnable scripts under `refclone/scripts/`.

---

## File Structure

```
refclone/
  .gitignore
  requirements.txt
  pytest.ini
  README.md
  src/
    refclone/
      __init__.py
      main.py               # FastAPI app + all endpoints
      gpu.py                 # CUDA/VRAM sanity-check utility
      audio_utils.py         # ffmpeg audio extraction + clean-segment selection
      voice/
        __init__.py
        enroll.py            # reference video -> stored voice sample -> voice_id
        synthesize.py        # VoiceSynthesizer (XTTS-v2 wrapper)
      lipsync/
        __init__.py
        musetalk.py           # subprocess wrapper around vendored MuseTalk
  scripts/
    check_gpu.py             # manual Phase 0 validation
    clone_and_speak.py       # Phase 1 CLI deliverable
    voice_only_mode.py       # Phase 2 CLI deliverable (full voice-only mode)
  tests/
    test_gpu.py
    test_audio_utils.py
    test_enroll.py
    test_synthesize.py
    test_musetalk.py
    test_api.py
    test_clone_and_speak.py
    test_voice_only_mode.py
  vendor/
    MuseTalk/                # cloned in Task 7, not committed (see .gitignore)
  data/
    voices/                  # enrolled voice samples, not committed
    outputs/                 # generated audio/video, not committed
```

---

### Task 1: Project Scaffold + Health Check

**Files:**
- Create: `refclone/.gitignore`
- Create: `refclone/requirements.txt`
- Create: `refclone/pytest.ini`
- Create: `refclone/src/refclone/__init__.py`
- Create: `refclone/src/refclone/main.py`
- Test: `refclone/tests/test_api.py`

**Interfaces:**
- Produces: FastAPI app instance `refclone.main.app`, importable and runnable via `uvicorn refclone.main:app`. All later tasks add endpoints to this same `app`.

- [ ] **Step 1: Create the project directory and non-code files**

Run:
```bash
mkdir -p "refclone/src/refclone" "refclone/scripts" "refclone/tests" "refclone/vendor" "refclone/data/voices" "refclone/data/outputs"
```

Create `refclone/.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/voices/*
data/outputs/*
!data/voices/.gitkeep
!data/outputs/.gitkeep
vendor/MuseTalk/
```

Create `refclone/data/voices/.gitkeep` and `refclone/data/outputs/.gitkeep` (empty files).

Create `refclone/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.9
pydub==0.25.1
numpy==1.26.4
torch>=2.1.0
coqui-tts==0.24.2
pytest==8.3.3
httpx==0.27.2
```

Create `refclone/pytest.ini`:
```ini
[pytest]
markers =
    gpu: tests that load real ML models and require a CUDA GPU (run manually with -m gpu)
testpaths = tests
```

- [ ] **Step 2: Set up the virtual environment and install dependencies**

Run:
```bash
cd refclone
python -m venv .venv
```

On Windows (Git Bash): `source .venv/Scripts/activate`
Then: `pip install -r requirements.txt`

Expected: install completes without errors. (`coqui-tts` and `torch` are large downloads; this step can take several minutes.)

- [ ] **Step 3: Create the empty package init**

Create `refclone/src/refclone/__init__.py` (empty file).

- [ ] **Step 4: Write the failing test for a health-check endpoint**

Create `refclone/tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from refclone.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `cd refclone && pip install -e . 2>/dev/null; PYTHONPATH=src pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.main'`

- [ ] **Step 6: Implement the FastAPI skeleton**

Create `refclone/src/refclone/main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="refclone")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 8: Init git and commit**

```bash
cd refclone
git init
git add .gitignore requirements.txt pytest.ini src tests data/voices/.gitkeep data/outputs/.gitkeep
git commit -m "feat: scaffold refclone FastAPI project with health check"
```

---

### Task 2: GPU/VRAM Sanity Check

**Files:**
- Create: `refclone/src/refclone/gpu.py`
- Create: `refclone/scripts/check_gpu.py`
- Test: `refclone/tests/test_gpu.py`

**Interfaces:**
- Produces: `refclone.gpu.get_device_info() -> dict` with keys `cuda_available: bool`, `device_name: str | None`, `vram_gb: float | None`. No other task consumes this directly, but it's the Phase 0 hardware-validation gate.

- [ ] **Step 1: Write the failing tests**

Create `refclone/tests/test_gpu.py`:
```python
from refclone import gpu


def test_get_device_info_when_cuda_available(monkeypatch):
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu.torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4070")

    class FakeProps:
        total_memory = 12 * 1024 ** 3

    monkeypatch.setattr(gpu.torch.cuda, "get_device_properties", lambda idx: FakeProps())

    info = gpu.get_device_info()

    assert info == {
        "cuda_available": True,
        "device_name": "NVIDIA RTX 4070",
        "vram_gb": 12.0,
    }


def test_get_device_info_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: False)

    info = gpu.get_device_info()

    assert info == {"cuda_available": False, "device_name": None, "vram_gb": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_gpu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.gpu'`

- [ ] **Step 3: Implement `gpu.py`**

Create `refclone/src/refclone/gpu.py`:
```python
import torch


def get_device_info() -> dict:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {"cuda_available": False, "device_name": None, "vram_gb": None}

    device_name = torch.cuda.get_device_name(0)
    total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(total_memory_bytes / (1024 ** 3), 1)

    return {
        "cuda_available": True,
        "device_name": device_name,
        "vram_gb": vram_gb,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_gpu.py -v`
Expected: PASS

- [ ] **Step 5: Write the manual validation script**

Create `refclone/scripts/check_gpu.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from refclone.gpu import get_device_info

if __name__ == "__main__":
    info = get_device_info()
    print(info)
    if not info["cuda_available"]:
        print("WARNING: no CUDA GPU detected. XTTS-v2 and MuseTalk will be very slow or unusable on CPU.")
    elif info["vram_gb"] < 8:
        print(f"WARNING: only {info['vram_gb']}GB VRAM detected; 8GB+ is the target for this pipeline.")
```

- [ ] **Step 6: Run the manual script against real hardware**

Run: `cd refclone && PYTHONPATH=src python scripts/check_gpu.py`
Expected: prints a dict with `cuda_available: True` and `vram_gb >= 8` on your machine. This is a manual hardware check, not part of the automated test suite.

- [ ] **Step 7: Commit**

```bash
cd refclone
git add src/refclone/gpu.py scripts/check_gpu.py tests/test_gpu.py
git commit -m "feat: add GPU/VRAM sanity check"
```

---

### Task 3: Audio Extraction & Clean-Segment Selection

**Files:**
- Create: `refclone/src/refclone/audio_utils.py`
- Test: `refclone/tests/test_audio_utils.py`

**Interfaces:**
- Produces:
  - `extract_audio_track(video_path: str, output_wav_path: str, sample_rate: int = 24000) -> str`
  - `select_clean_segment(wav_path: str, output_wav_path: str, min_seconds: float = 6.0, max_seconds: float = 30.0, silence_thresh_db: int = -40) -> str` — raises `ValueError` if no usable non-silent audio is found.
- Consumed by: Task 4 (`voice/enroll.py`).

- [ ] **Step 1: Write the failing tests**

Create `refclone/tests/test_audio_utils.py`:
```python
import subprocess
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from refclone.audio_utils import extract_audio_track, select_clean_segment


def test_extract_audio_track_invokes_ffmpeg_with_expected_args(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, check, capture_output):
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_path = str(tmp_path / "out.wav")
    result = extract_audio_track("input.mp4", output_path, sample_rate=24000)

    assert result == output_path
    assert captured["cmd"] == [
        "ffmpeg", "-y", "-i", "input.mp4", "-ac", "1", "-ar", "24000", output_path,
    ]
    assert captured["check"] is True


def test_select_clean_segment_extracts_tone_from_silence(tmp_path):
    tone = Sine(440).to_audio_segment(duration=8000)
    audio = AudioSegment.silent(duration=1000) + tone + AudioSegment.silent(duration=1000)
    input_path = tmp_path / "input.wav"
    audio.export(input_path, format="wav")

    output_path = tmp_path / "clean.wav"
    result = select_clean_segment(str(input_path), str(output_path), min_seconds=6.0, max_seconds=30.0)

    assert result == str(output_path)
    segment = AudioSegment.from_wav(output_path)
    assert 6.0 <= len(segment) / 1000 <= 30.0


def test_select_clean_segment_raises_on_pure_silence(tmp_path):
    audio = AudioSegment.silent(duration=5000)
    input_path = tmp_path / "silent.wav"
    audio.export(input_path, format="wav")

    with pytest.raises(ValueError):
        select_clean_segment(str(input_path), str(tmp_path / "out.wav"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_audio_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.audio_utils'`

- [ ] **Step 3: Implement `audio_utils.py`**

Create `refclone/src/refclone/audio_utils.py`:
```python
import subprocess

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def extract_audio_track(video_path: str, output_wav_path: str, sample_rate: int = 24000) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sample_rate), output_wav_path],
        check=True,
        capture_output=True,
    )
    return output_wav_path


def select_clean_segment(
    wav_path: str,
    output_wav_path: str,
    min_seconds: float = 6.0,
    max_seconds: float = 30.0,
    silence_thresh_db: int = -40,
) -> str:
    audio = AudioSegment.from_wav(wav_path)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=300, silence_thresh=silence_thresh_db)

    if not nonsilent_ranges:
        raise ValueError(f"No non-silent audio found in {wav_path}")

    start_ms, end_ms = max(nonsilent_ranges, key=lambda r: r[1] - r[0])
    min_ms, max_ms = min_seconds * 1000, max_seconds * 1000
    duration_ms = end_ms - start_ms

    if duration_ms < min_ms:
        pad = (min_ms - duration_ms) / 2
        start_ms = max(0, start_ms - pad)
        end_ms = min(len(audio), end_ms + pad)
    elif duration_ms > max_ms:
        end_ms = start_ms + max_ms

    segment = audio[start_ms:end_ms]
    if len(segment) < min_ms:
        raise ValueError(f"Clean segment too short after padding: {len(segment)}ms < {min_ms}ms")

    segment.export(output_wav_path, format="wav")
    return output_wav_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_audio_utils.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd refclone
git add src/refclone/audio_utils.py tests/test_audio_utils.py
git commit -m "feat: add ffmpeg audio extraction and clean-segment selection"
```

---

### Task 4: Voice Enrollment (`enroll_voice` + `/clone-voice`)

**Files:**
- Create: `refclone/src/refclone/voice/__init__.py`
- Create: `refclone/src/refclone/voice/enroll.py`
- Modify: `refclone/src/refclone/main.py`
- Test: `refclone/tests/test_enroll.py`
- Test: `refclone/tests/test_api.py`

**Interfaces:**
- Consumes: `extract_audio_track`, `select_clean_segment` from Task 3.
- Produces: `enroll_voice(reference_video_path: str, voices_dir: Path = VOICES_DIR) -> str` returning a `voice_id`. Stores the cleaned sample at `data/voices/{voice_id}.wav`. Consumed by Task 5 (`/synthesize`, CLI scripts).
- Endpoint: `POST /clone-voice` — multipart file upload field `video`, returns `{"voice_id": str}`.

- [ ] **Step 1: Write the failing test for `enroll_voice`**

Create `refclone/src/refclone/voice/__init__.py` (empty file).

Create `refclone/tests/test_enroll.py`:
```python
from pathlib import Path

from refclone.voice import enroll


def test_enroll_voice_orchestrates_extraction_and_cleans_up(monkeypatch, tmp_path):
    calls = []

    def fake_extract(video_path, output_wav_path, sample_rate=24000):
        calls.append(("extract", video_path, output_wav_path))
        Path(output_wav_path).write_bytes(b"raw")
        return output_wav_path

    def fake_select(wav_path, output_wav_path, min_seconds=6.0, max_seconds=30.0, silence_thresh_db=-40):
        calls.append(("select", wav_path, output_wav_path))
        Path(output_wav_path).write_bytes(b"clean")
        return output_wav_path

    monkeypatch.setattr(enroll, "extract_audio_track", fake_extract)
    monkeypatch.setattr(enroll, "select_clean_segment", fake_select)

    voice_id = enroll.enroll_voice("reference.mp4", voices_dir=tmp_path)

    clean_path = tmp_path / f"{voice_id}.wav"
    raw_path = tmp_path / f"{voice_id}_raw.wav"

    assert clean_path.exists()
    assert clean_path.read_bytes() == b"clean"
    assert not raw_path.exists()  # raw intermediate file is cleaned up
    assert calls[0][0] == "extract"
    assert calls[1][0] == "select"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_enroll.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.voice.enroll'`

- [ ] **Step 3: Implement `enroll.py`**

Create `refclone/src/refclone/voice/enroll.py`:
```python
import uuid
from pathlib import Path

from refclone.audio_utils import extract_audio_track, select_clean_segment

VOICES_DIR = Path("data/voices")


def enroll_voice(reference_video_path: str, voices_dir: Path = VOICES_DIR) -> str:
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_id = str(uuid.uuid4())
    raw_wav = voices_dir / f"{voice_id}_raw.wav"
    clean_wav = voices_dir / f"{voice_id}.wav"

    extract_audio_track(reference_video_path, str(raw_wav))
    select_clean_segment(str(raw_wav), str(clean_wav))
    raw_wav.unlink()

    return voice_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_enroll.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `/clone-voice`**

Add to `refclone/tests/test_api.py`:
```python
import io


def test_clone_voice_endpoint_returns_voice_id(monkeypatch):
    monkeypatch.setattr("refclone.main.enroll_voice", lambda path, **kwargs: "fixed-voice-id")

    response = client.post(
        "/clone-voice",
        files={"video": ("reference.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == {"voice_id": "fixed-voice-id"}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: FAIL — 404 or `AttributeError` (no `/clone-voice` route, no `enroll_voice` imported in `main`)

- [ ] **Step 7: Add the endpoint to `main.py`**

Modify `refclone/src/refclone/main.py`:
```python
import os
import tempfile

from fastapi import FastAPI, UploadFile

from refclone.voice.enroll import enroll_voice

app = FastAPI(title="refclone")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/clone-voice")
async def clone_voice_endpoint(video: UploadFile) -> dict:
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        voice_id = enroll_voice(tmp_path)
    finally:
        os.remove(tmp_path)

    return {"voice_id": voice_id}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: PASS (both `test_health_check_returns_ok` and `test_clone_voice_endpoint_returns_voice_id`)

- [ ] **Step 9: Commit**

```bash
cd refclone
git add src/refclone/voice src/refclone/main.py tests/test_enroll.py tests/test_api.py
git commit -m "feat: add voice enrollment and /clone-voice endpoint"
```

---

### Task 5: XTTS-v2 Voice Synthesis (`VoiceSynthesizer` + `/synthesize`)

**Files:**
- Create: `refclone/src/refclone/voice/synthesize.py`
- Modify: `refclone/src/refclone/main.py`
- Test: `refclone/tests/test_synthesize.py`
- Test: `refclone/tests/test_api.py`

**Interfaces:**
- Produces: `VoiceSynthesizer` class with `synthesize(self, text: str, speaker_wav_path: str, language: str, output_path: str) -> str`. Consumed by Task 6 and Task 8 CLI scripts, and by the `/synthesize` endpoint.
- Endpoint: `POST /synthesize` with JSON body `{"voice_id": str, "text": str, "language": str = "en"}`, returns `{"audio_path": str}`. Returns 404 if `voice_id` is unknown.

- [ ] **Step 1: Write the failing unit test for `VoiceSynthesizer`**

Create `refclone/tests/test_synthesize.py`:
```python
from refclone.voice.synthesize import VoiceSynthesizer


def test_synthesize_calls_underlying_tts_with_expected_args(monkeypatch, tmp_path):
    calls = []

    class FakeTTS:
        def to(self, device):
            return self

        def tts_to_file(self, text, speaker_wav, language, file_path):
            calls.append((text, speaker_wav, language, file_path))

    monkeypatch.setattr(
        "refclone.voice.synthesize._load_tts_backend",
        lambda model_name: FakeTTS(),
    )

    synthesizer = VoiceSynthesizer()
    output_path = str(tmp_path / "out.wav")
    result = synthesizer.synthesize("hello world", "speaker.wav", "en", output_path)

    assert result == output_path
    assert calls == [("hello world", "speaker.wav", "en", output_path)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_synthesize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.voice.synthesize'`

- [ ] **Step 3: Implement `synthesize.py`**

Create `refclone/src/refclone/voice/synthesize.py`:
```python
def _load_tts_backend(model_name: str):
    from TTS.api import TTS  # imported lazily so unit tests don't need the real model loaded

    return TTS(model_name)


class VoiceSynthesizer:
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
    ):
        self._tts = _load_tts_backend(model_name).to(device)

    def synthesize(self, text: str, speaker_wav_path: str, language: str, output_path: str) -> str:
        self._tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_path,
        )
        return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_synthesize.py -v`
Expected: PASS

- [ ] **Step 5: Write the real-model integration test (manual, GPU-gated)**

Append to `refclone/tests/test_synthesize.py`:
```python
import pytest


@pytest.mark.gpu
def test_real_xtts_synthesis_produces_audio_file(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    speaker_wav = tmp_path / "speaker.wav"
    Sine(220).to_audio_segment(duration=8000).export(speaker_wav, format="wav")

    synthesizer = VoiceSynthesizer()
    output_path = tmp_path / "out.wav"
    synthesizer.synthesize("This is a test of voice cloning.", str(speaker_wav), "en", str(output_path))

    assert output_path.exists()
    assert AudioSegment.from_wav(output_path).duration_seconds > 0
```

This test is skipped by default (not run in Step 4 or 6's plain `pytest` invocations). Run it manually once, after Task 5, to confirm XTTS-v2 actually works on your GPU:

Run: `cd refclone && PYTHONPATH=src pytest tests/test_synthesize.py -v -m gpu`
Expected: PASS (will download the ~2GB XTTS-v2 model on first run — this can take several minutes).

- [ ] **Step 6: Write the failing test for `/synthesize`**

Add to `refclone/tests/test_api.py`:
```python
def test_synthesize_endpoint_writes_audio_and_returns_path(tmp_path, monkeypatch):
    from refclone.main import app, get_synthesizer

    monkeypatch.setattr("refclone.main.VOICES_DIR", tmp_path)
    (tmp_path / "known-voice.wav").write_bytes(b"fake speaker sample")

    class FakeSynthesizer:
        def synthesize(self, text, speaker_wav_path, language, output_path):
            from pathlib import Path
            Path(output_path).write_bytes(b"fake synthesized audio")
            return output_path

    app.dependency_overrides[get_synthesizer] = lambda: FakeSynthesizer()
    try:
        response = client.post(
            "/synthesize",
            json={"voice_id": "known-voice", "text": "hello", "language": "en"},
        )
    finally:
        app.dependency_overrides.pop(get_synthesizer, None)

    assert response.status_code == 200
    audio_path = response.json()["audio_path"]
    assert Path(audio_path).exists()


def test_synthesize_endpoint_404s_on_unknown_voice(tmp_path, monkeypatch):
    from refclone.main import app

    monkeypatch.setattr("refclone.main.VOICES_DIR", tmp_path)

    response = client.post(
        "/synthesize",
        json={"voice_id": "no-such-voice", "text": "hello", "language": "en"},
    )

    assert response.status_code == 404
```

Add `from pathlib import Path` to the top of `refclone/tests/test_api.py` if not already present.

- [ ] **Step 7: Run test to verify it fails**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: FAIL — no `/synthesize` route, no `get_synthesizer` in `refclone.main`

- [ ] **Step 8: Add the endpoint to `main.py`**

Modify `refclone/src/refclone/main.py` (add imports and new code, keep existing `/health` and `/clone-voice`):
```python
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from refclone.voice.enroll import enroll_voice
from refclone.voice.synthesize import VoiceSynthesizer

app = FastAPI(title="refclone")

VOICES_DIR = Path("data/voices")
OUTPUTS_DIR = Path("data/outputs")

_synthesizer: VoiceSynthesizer | None = None


def get_synthesizer() -> VoiceSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = VoiceSynthesizer()
    return _synthesizer


class SynthesizeRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/clone-voice")
async def clone_voice_endpoint(video: UploadFile) -> dict:
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        voice_id = enroll_voice(tmp_path, voices_dir=VOICES_DIR)
    finally:
        os.remove(tmp_path)

    return {"voice_id": voice_id}


@app.post("/synthesize")
async def synthesize_endpoint(
    req: SynthesizeRequest, synthesizer: VoiceSynthesizer = Depends(get_synthesizer)
) -> dict:
    speaker_wav = VOICES_DIR / f"{req.voice_id}.wav"
    if not speaker_wav.exists():
        raise HTTPException(status_code=404, detail=f"Unknown voice_id: {req.voice_id}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.wav"
    synthesizer.synthesize(req.text, str(speaker_wav), req.language, str(output_path))

    return {"audio_path": str(output_path)}
```

Note: Step 6's test for `/clone-voice` (Task 4) referenced `enroll_voice(path, **kwargs)` via monkeypatch — this still passes since `voices_dir` is passed as a keyword argument.

- [ ] **Step 9: Run test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 10: Commit**

```bash
cd refclone
git add src/refclone/voice/synthesize.py src/refclone/main.py tests/test_synthesize.py tests/test_api.py
git commit -m "feat: add XTTS-v2 voice synthesis and /synthesize endpoint"
```

---

### Task 6: CLI Script `clone_and_speak.py` (Phase 1 Deliverable)

**Files:**
- Create: `refclone/scripts/clone_and_speak.py`
- Test: `refclone/tests/test_clone_and_speak.py`

**Interfaces:**
- Consumes: `enroll_voice` (Task 4), `VoiceSynthesizer` (Task 5).
- Produces: `build_parser() -> argparse.ArgumentParser`, `enforce_consent(args) -> None` (raises `SystemExit` if `args.i_have_consent` is falsy), `run(args, synthesizer: VoiceSynthesizer | None = None) -> str` (returns output path). Reused by Task 8's consent-gate pattern.

- [ ] **Step 1: Write the failing tests**

Create `refclone/tests/test_clone_and_speak.py`:
```python
from pathlib import Path

import pytest

import scripts.clone_and_speak as cli


def test_enforce_consent_raises_without_flag():
    args = cli.build_parser().parse_args(
        ["--reference-video", "ref.mp4", "--script-file", "s.txt", "--output", "out.wav"]
    )
    with pytest.raises(SystemExit):
        cli.enforce_consent(args)


def test_enforce_consent_passes_with_flag():
    args = cli.build_parser().parse_args(
        [
            "--reference-video", "ref.mp4",
            "--script-file", "s.txt",
            "--output", "out.wav",
            "--i-have-consent",
        ]
    )
    cli.enforce_consent(args)  # should not raise


def test_run_orchestrates_enroll_and_synthesize(monkeypatch, tmp_path):
    script_file = tmp_path / "script.txt"
    script_file.write_text("Hello from the script.")
    output_path = tmp_path / "out.wav"

    monkeypatch.setattr(cli, "enroll_voice", lambda video_path: "voice-123")

    calls = []

    class FakeSynthesizer:
        def synthesize(self, text, speaker_wav_path, language, output_path):
            calls.append((text, speaker_wav_path, language, output_path))
            Path(output_path).write_bytes(b"audio")
            return output_path

    args = cli.build_parser().parse_args(
        [
            "--reference-video", "ref.mp4",
            "--script-file", str(script_file),
            "--output", str(output_path),
            "--i-have-consent",
        ]
    )

    result = cli.run(args, synthesizer=FakeSynthesizer())

    assert result == str(output_path)
    assert calls == [("Hello from the script.", "data/voices/voice-123.wav", "en", str(output_path))]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refclone && PYTHONPATH=src:. pytest tests/test_clone_and_speak.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.clone_and_speak'`

- [ ] **Step 3: Implement the CLI script**

Create `refclone/scripts/__init__.py` (empty file, makes `scripts` importable as a package for tests).

Create `refclone/scripts/clone_and_speak.py`:
```python
import argparse
from pathlib import Path

from refclone.voice.enroll import enroll_voice
from refclone.voice.synthesize import VoiceSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone a voice from a reference video and speak a script with it."
    )
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--script-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="Confirms you own or have explicit rights to clone the voice in --reference-video.",
    )
    return parser


def enforce_consent(args) -> None:
    if not args.i_have_consent:
        raise SystemExit(
            "Refusing to run: pass --i-have-consent to confirm you own or have "
            "explicit rights to clone the voice in the reference video."
        )


def run(args, synthesizer: VoiceSynthesizer | None = None) -> str:
    enforce_consent(args)

    voice_id = enroll_voice(args.reference_video)
    text = Path(args.script_file).read_text()
    synth = synthesizer or VoiceSynthesizer()
    synth.synthesize(text, f"data/voices/{voice_id}.wav", args.language, args.output)

    return args.output


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    print(f"Voice enrolled and script synthesized.")
    print(f"Output written to: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refclone && PYTHONPATH=src:. pytest tests/test_clone_and_speak.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd refclone
git add scripts/__init__.py scripts/clone_and_speak.py tests/test_clone_and_speak.py
git commit -m "feat: add clone_and_speak CLI (Phase 1 deliverable)"
```

---

### Task 7: MuseTalk Lip-Sync Wrapper (`run_lipsync` + `/lip-sync`)

**Files:**
- Create: `refclone/src/refclone/lipsync/__init__.py`
- Create: `refclone/src/refclone/lipsync/musetalk.py`
- Modify: `refclone/src/refclone/main.py`
- Test: `refclone/tests/test_musetalk.py`
- Test: `refclone/tests/test_api.py`

**Interfaces:**
- Produces: `run_lipsync(video_path: str, audio_path: str, output_path: str, musetalk_root: Path = MUSETALK_ROOT) -> str`. Consumed by Task 8's CLI script and the `/lip-sync` endpoint.
- Endpoint: `POST /lip-sync` with JSON body `{"video_path": str, "audio_path": str}`, returns `{"video_path": str}`.

**Before writing code — vendor MuseTalk:**

```bash
cd refclone/vendor
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
pip install -r requirements.txt
```

Follow MuseTalk's README to download its model checkpoints (whisper, VAE, UNet, DWPose weights) into the paths it expects. **Confirm the exact `scripts.inference` CLI flags in the README at the commit you just cloned** — MuseTalk's interface has changed across releases, and `run_lipsync` below is written against the documented `--video_path` / `--audio_path` / `--result_dir` flow. If your cloned version differs, adjust the `subprocess.run` argument list in Step 3 to match — the tests in this task mock `subprocess.run`, so they stay green regardless; only the manual smoke test in Task 8 depends on getting the real flags right.

- [ ] **Step 1: Write the failing tests**

Create `refclone/src/refclone/lipsync/__init__.py` (empty file).

Create `refclone/tests/test_musetalk.py`:
```python
import subprocess
from pathlib import Path

import pytest

from refclone.lipsync.musetalk import run_lipsync


def test_run_lipsync_invokes_subprocess_with_expected_args(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd, check, capture_output):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        result_dir = Path(tmp_path) / "musetalk_tmp"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.mp4").write_bytes(b"fake output video")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_path = tmp_path / "final.mp4"
    result = run_lipsync("clip.mp4", "audio.wav", str(output_path), musetalk_root=tmp_path / "MuseTalk")

    assert result == str(output_path)
    assert output_path.exists()
    assert captured["cwd"] == str(tmp_path / "MuseTalk")
    assert "clip.mp4" in captured["cmd"]
    assert "audio.wav" in captured["cmd"]


def test_run_lipsync_raises_when_no_output_produced(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, check, capture_output):
        result_dir = Path(tmp_path) / "musetalk_tmp"
        result_dir.mkdir(parents=True, exist_ok=True)
        # no .mp4 written -- simulates MuseTalk failing silently

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        run_lipsync(
            "clip.mp4", "audio.wav", str(tmp_path / "final.mp4"), musetalk_root=tmp_path / "MuseTalk"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_musetalk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'refclone.lipsync.musetalk'`

- [ ] **Step 3: Implement `musetalk.py`**

Create `refclone/src/refclone/lipsync/musetalk.py`:
```python
import shutil
import subprocess
from pathlib import Path

MUSETALK_ROOT = Path("vendor/MuseTalk")


def run_lipsync(
    video_path: str, audio_path: str, output_path: str, musetalk_root: Path = MUSETALK_ROOT
) -> str:
    result_dir = Path(output_path).parent / "musetalk_tmp"
    result_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "python", "-m", "scripts.inference",
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--result_dir", str(result_dir),
        ],
        cwd=str(musetalk_root),
        check=True,
        capture_output=True,
    )

    produced = _find_output_video(result_dir)
    shutil.move(str(produced), output_path)
    return output_path


def _find_output_video(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"MuseTalk produced no .mp4 output in {result_dir}")
    return candidates[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_musetalk.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing test for `/lip-sync`**

Add to `refclone/tests/test_api.py`:
```python
def test_lipsync_endpoint_returns_video_path(monkeypatch):
    def fake_run_lipsync(video_path, audio_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"fake output")
        return output_path

    monkeypatch.setattr("refclone.main.run_lipsync", fake_run_lipsync)

    response = client.post(
        "/lip-sync", json={"video_path": "clip.mp4", "audio_path": "audio.wav"}
    )

    assert response.status_code == 200
    video_path = response.json()["video_path"]
    assert Path(video_path).exists()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: FAIL — no `/lip-sync` route

- [ ] **Step 7: Add the endpoint to `main.py`**

Add to `refclone/src/refclone/main.py` (alongside existing imports and routes):
```python
from refclone.lipsync.musetalk import run_lipsync


class LipSyncRequest(BaseModel):
    video_path: str
    audio_path: str


@app.post("/lip-sync")
async def lipsync_endpoint(req: LipSyncRequest) -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{uuid.uuid4()}.mp4"
    run_lipsync(req.video_path, req.audio_path, str(output_path))
    return {"video_path": str(output_path)}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd refclone && PYTHONPATH=src pytest tests/test_api.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 9: Commit**

```bash
cd refclone
git add src/refclone/lipsync src/refclone/main.py tests/test_musetalk.py tests/test_api.py
git commit -m "feat: add MuseTalk lip-sync wrapper and /lip-sync endpoint"
```

---

### Task 8: CLI Script `voice_only_mode.py` (Phase 2 Deliverable — Full Voice-Only Mode)

**Files:**
- Create: `refclone/scripts/voice_only_mode.py`
- Test: `refclone/tests/test_voice_only_mode.py`

**Interfaces:**
- Consumes: `enroll_voice` (Task 4), `VoiceSynthesizer` (Task 5), `run_lipsync` (Task 7).
- Produces: end-to-end CLI: reference video + your clip + script → final lip-synced video with the reference speaker's cloned voice.

- [ ] **Step 1: Write the failing tests**

Create `refclone/tests/test_voice_only_mode.py`:
```python
from pathlib import Path

import pytest

import scripts.voice_only_mode as cli


def test_enforce_consent_raises_without_flag():
    args = cli.build_parser().parse_args(
        [
            "--reference-video", "ref.mp4",
            "--your-clip", "clip.mp4",
            "--script-file", "s.txt",
            "--output", "out.mp4",
        ]
    )
    with pytest.raises(SystemExit):
        cli.enforce_consent(args)


def test_run_orchestrates_enroll_synthesize_and_lipsync(monkeypatch, tmp_path):
    script_file = tmp_path / "script.txt"
    script_file.write_text("Hello from the script.")
    output_path = tmp_path / "final.mp4"

    monkeypatch.setattr(cli, "enroll_voice", lambda video_path: "voice-123")

    synth_calls = []

    class FakeSynthesizer:
        def synthesize(self, text, speaker_wav_path, language, output_path):
            synth_calls.append((text, speaker_wav_path, language, output_path))
            Path(output_path).write_bytes(b"audio")
            return output_path

    lipsync_calls = []

    def fake_run_lipsync(video_path, audio_path, output_path, **kwargs):
        lipsync_calls.append((video_path, audio_path, output_path))
        Path(output_path).write_bytes(b"video")
        return output_path

    monkeypatch.setattr(cli, "run_lipsync", fake_run_lipsync)

    args = cli.build_parser().parse_args(
        [
            "--reference-video", "ref.mp4",
            "--your-clip", "clip.mp4",
            "--script-file", str(script_file),
            "--output", str(output_path),
            "--i-have-consent",
        ]
    )

    result = cli.run(args, synthesizer=FakeSynthesizer())

    assert result == str(output_path)
    assert output_path.exists()
    assert synth_calls[0][0] == "Hello from the script."
    assert synth_calls[0][1] == "data/voices/voice-123.wav"
    assert lipsync_calls[0][0] == "clip.mp4"
    assert lipsync_calls[0][2] == str(output_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd refclone && PYTHONPATH=src:. pytest tests/test_voice_only_mode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.voice_only_mode'`

- [ ] **Step 3: Implement the CLI script**

Create `refclone/scripts/voice_only_mode.py`:
```python
import argparse
from pathlib import Path

from refclone.voice.enroll import enroll_voice
from refclone.voice.synthesize import VoiceSynthesizer
from refclone.lipsync.musetalk import run_lipsync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Voice-only clone mode: keep your clip's visuals, speak your script "
            "in the reference video's cloned voice, and re-sync lips to match."
        )
    )
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--your-clip", required=True)
    parser.add_argument("--script-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="Confirms you own or have explicit rights to clone the voice in --reference-video.",
    )
    return parser


def enforce_consent(args) -> None:
    if not args.i_have_consent:
        raise SystemExit(
            "Refusing to run: pass --i-have-consent to confirm you own or have "
            "explicit rights to clone the voice in the reference video."
        )


def run(args, synthesizer: VoiceSynthesizer | None = None) -> str:
    enforce_consent(args)

    voice_id = enroll_voice(args.reference_video)
    text = Path(args.script_file).read_text()

    audio_output = Path("data/outputs") / f"{voice_id}_script.wav"
    audio_output.parent.mkdir(parents=True, exist_ok=True)

    synth = synthesizer or VoiceSynthesizer()
    synth.synthesize(text, f"data/voices/{voice_id}.wav", args.language, str(audio_output))

    run_lipsync(args.your_clip, str(audio_output), args.output)

    return args.output


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    print(f"Voice-only clone complete. Output written to: {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd refclone && PYTHONPATH=src:. pytest tests/test_voice_only_mode.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full automated test suite**

Run: `cd refclone && PYTHONPATH=src:. pytest tests/ -v`
Expected: all tests pass except any marked `gpu` (those are excluded by default only if you configure `addopts = -m "not gpu"` in `pytest.ini` — otherwise they'll also attempt to run; if you haven't run the manual GPU check from Task 5 Step 5 yet, run `pytest tests/ -v -m "not gpu"` here instead).

- [ ] **Step 6: Commit**

```bash
cd refclone
git add scripts/voice_only_mode.py tests/test_voice_only_mode.py
git commit -m "feat: add voice_only_mode CLI (Phase 2 deliverable)"
```

- [ ] **Step 7: Manual end-to-end smoke test with real, consented media**

This is a manual validation step, not an automated test. Using a reference video of yourself (or another consenting subject) and a separate clip of yourself:

Run:
```bash
cd refclone
PYTHONPATH=src:. python scripts/voice_only_mode.py \
  --reference-video /path/to/your/reference.mp4 \
  --your-clip /path/to/your/clip.mp4 \
  --script-file /path/to/script.txt \
  --output data/outputs/smoke_test.mp4 \
  --i-have-consent
```

Expected: `data/outputs/smoke_test.mp4` is produced. Watch/listen to it — the visuals should be your clip's, the voice should sound like the reference video's speaker, and the mouth movement should roughly match the new script. If MuseTalk's real CLI flags differ from Task 7's implementation, this step will surface it with a concrete subprocess error — adjust `run_lipsync`'s command list accordingly and re-run.

---

## Self-Review Notes

- **Spec coverage**: Phase 0 (setup) → Task 1-2. Phase 1 (voice cloning end-to-end) → Tasks 3-6. Phase 2 (lip-sync, ships voice-only mode) → Tasks 7-8. Consent-gate principle from `planforreferencevideo.md` §2 → enforced via `--i-have-consent` in both CLI scripts (interim measure, full DB-backed gate is Phase 5). Local-only/no-cloud-APIs constraint → XTTS-v2 and MuseTalk both run locally, no external API calls anywhere in this slice.
- **Deferred to later plans** (per `planforreferencevideo.md`'s remaining phases, not in scope here): LivePortrait face reenactment (Phase 3), style transfer (Phase 4), Next.js UI + SQLite job queue + DB-backed consent gate (Phase 5), Remotion assembly/captions (Phase 6), hardening — input validation for "no face detected," VRAM serialization across concurrent jobs, output quality gates (Phase 7).
- **Type/signature consistency checked**: `enroll_voice(reference_video_path: str, voices_dir: Path = VOICES_DIR) -> str` used identically in Tasks 4, 6, 8. `VoiceSynthesizer.synthesize(text, speaker_wav_path, language, output_path) -> str` signature matches across Task 5's endpoint, Task 6, and Task 8. `run_lipsync(video_path, audio_path, output_path, musetalk_root=...) -> str` matches across Task 7's endpoint and Task 8.
