# Reference-Video-Driven Video Generator — Plan

## 1. Idea

Give the tool three inputs:
1. A **reference video** — supplies face/mannerisms, voice, and/or editing style to clone/mimic.
2. **Your own video clip** — the footage/performance to reuse or drive the output.
3. A **script** — the words the final video should say.

The tool combines these into a new video. Three independent, toggleable modes cover the different ways "based on the reference video" can be interpreted:

- **Voice-only clone** — keep your clip's visuals, replace/add narration in the reference speaker's cloned voice reading the script, with lip-sync correction.
- **Full clone (face + voice + mannerisms)** — the reference person's face/identity performs your clip's motion/timing, speaking the script in their cloned voice.
- **Style-only transfer** — apply the reference video's editing rhythm (cuts, pacing, camera moves) to your own footage; no face/voice cloning involved.

All three can be combined (e.g. full clone + style transfer) or used standalone.

## 2. Consent Scope (hard constraint)

**This tool is scoped to consenting subjects only**: yourself, or third parties (clients, actors, licensed footage) who have explicitly given rights to clone their face/voice. It is **not** designed or intended to clone public figures, influencers, or any other real person from arbitrary internet video without their consent.

This is enforced in the product, not just as a guideline:
- Every reference video import requires a consent record: subject = `Myself` or `Licensed/consenting third party` (name + how rights were obtained), plus a confirmation checkbox.
- The consent record (subject label, rights note, timestamp, file hash of the reference video) is stored in SQLite alongside the video.
- The job runner refuses to start any job against a reference video with no consent record — enforced server-side (DB constraint / runner check), not just a frontend checkbox that could be bypassed.
- No "paste a URL and clone whoever's in it" shortcut. Reference videos must be explicitly imported files, preserving the friction of an active "I attest I have rights to this" step.

## 3. Architecture

Three local components, no cloud APIs (matches the `vidgen` project's local-only philosophy: local Ollama for LLM, local Kokoro for TTS).

```
┌─────────────────────┐      REST       ┌──────────────────────────┐
│   app (Next.js)      │ ───────────────▶│  inference-server          │
│  local UI, no deploy │                  │  (Python/FastAPI, GPU)    │
│  - upload ref video  │                  │  - XTTS-v2 (voice clone)  │
│  - upload your clip  │                  │  - LivePortrait (face)    │
│  - paste script      │                  │  - MuseTalk (lip-sync)    │
│  - pick mode(s)       │                  │  - OpenCV style analysis  │
│  - consent checkbox  │                  └──────────────────────────┘
│  - job progress/     │
│    preview/download  │
└─────────┬────────────┘
          │
          ▼
   SQLite job store  ◀──── job runner (Node, polls SQLite like
   (jobs, consent          vidgen's poller polls Supabase; serializes
   records, stages)        GPU-heavy stages to avoid VRAM contention)
          │
          ▼
   assembly (Remotion, reused from vidgen)
   - captions, branding, aspect-ratio crop
   - final MP4 export
```

**Why this shape**: mirrors what already works in `vidgen` — Kokoro runs as an isolated local service today; `inference-server` follows the same pattern for the heavier ML models. Reuses the existing Remotion pipeline for captions/branding instead of rebuilding that.

## 4. Components

### 4.1 `inference-server` (Python/FastAPI, GPU)
- `POST /clone-voice` — reference audio in, speaker embedding out (auto-extracts the longest clean 6–30s segment from the reference video's audio track).
- `POST /synthesize` — script + speaker embedding in, cloned-voice audio out. Model: **XTTS-v2** (zero-shot voice cloning, multilingual, runs on 8GB+ VRAM).
- `POST /reenact-face` — reference video/still (identity) + your clip (driving motion) in, reenacted video out. Model: **LivePortrait**.
- `POST /lip-sync` — video + new audio in, lip-synced video out. Model: **MuseTalk**.
- `POST /analyze-style` — reference video in, style profile JSON out (cut timestamps, shot durations, camera-motion tags). Tooling: OpenCV + PySceneDetect.
- Input validation at each endpoint (face detected? audio clear?) before GPU work starts.

### 4.2 `app` (Next.js, local only)
- Upload reference video + your clip, paste script, consent form, mode toggles, job progress view, preview/download.
- Talks to `inference-server` and the SQLite job store.

### 4.3 Job runner (Node)
- Polls SQLite (same pattern as `vidgen`'s poller polling Supabase, but fully local — no external service needed).
- Sequences multi-stage pipelines per job, serializes GPU-heavy stages across concurrent jobs (LivePortrait + XTTS + MuseTalk together can exceed 8GB VRAM if run concurrently).
- Marks jobs `failed` with the specific stage + message on error; keeps partial outputs (e.g. already-generated cloned audio) available for reuse.

### 4.4 `assembly` (reused from `vidgen`)
- Remotion composition adds captions/branding/aspect-ratio crop to the generated video+audio, same as `vidgen`'s existing render step.
- For style-only mode, Remotion also drives the re-cut/re-pace of your footage per the extracted style profile.

## 5. Pipeline Modes — Data Flow

### Voice-only clone
```
reference video ─▶ /clone-voice ─▶ speaker embedding
script ──────────────────────────▶ /synthesize ─▶ cloned-voice audio
your clip + cloned-voice audio ──▶ /lip-sync ─▶ output video
```
Output = your visuals, their voice, your script.

### Full clone (face + voice + mannerisms)
```
reference video ─▶ /clone-voice ─▶ speaker embedding
script ──────────────────────────▶ /synthesize ─▶ cloned-voice audio
reference video (identity) + your clip (driving motion) ─▶ /reenact-face ─▶ reenacted video
reenacted video + cloned-voice audio ─▶ /lip-sync ─▶ output video
```
Output = their face/identity, their voice, your script, your performance timing.

### Style-only transfer
```
reference video ─▶ /analyze-style ─▶ style profile JSON
your clip + style profile + (your original or cloned audio) ─▶ Remotion re-edit ─▶ output video
```
No face/voice cloning; can stack with either mode above or run standalone.

All modes converge on the `assembly` step for captions/branding.

## 6. Error Handling & Validation

- **Input validation up front**: reject early (before GPU work) if no face is detected in the reference video or the clip, or if the audio track is unusable — clear error message at submission time, not failure 4 minutes into a run.
- **Per-stage failure isolation**: job runner catches errors per stage, marks the job `failed` with which stage + why, and preserves any completed intermediate outputs (e.g. cloned audio survives a later face-reenactment failure) so they can be reused without re-running the whole job.
- **VRAM contention**: only one GPU-heavy stage executes at a time across all jobs (serialized by the job runner), since XTTS-v2 + LivePortrait + MuseTalk running concurrently can exceed 8GB VRAM.
- **Quality gate before assembly**: automatic sanity check (face present across sampled output frames, audio/video duration match) before handing off to Remotion, so broken generations aren't silently marked "complete."

## 7. Testing/Validation Approach

- ML pipeline stages: validated manually against a small fixed set of consented test reference videos/clips/scripts (yours), checked by eye/ear after any model or pipeline change — not meaningfully unit-testable.
- Job-runner state machine (job created → stage transitions → completed/failed) and FastAPI endpoint contracts: real automated tests, since this logic is deterministic and prone to silent regressions.

## 8. Phased Build Plan

| Phase | Deliverable |
|---|---|
| 0 | Project setup: new folder (sibling to `vidgen`/`avatar-gen`), Python venv + FastAPI skeleton, CUDA sanity check, pull XTTS-v2/LivePortrait/MuseTalk weights. |
| 1 | Voice cloning end-to-end: `/clone-voice` + `/synthesize`. Validate on your own voice. |
| 2 | Lip-sync on existing footage: `/lip-sync` endpoint re-syncs your clip to new cloned audio. **Ships "voice-only" mode.** |
| 3 | Face reenactment: `/reenact-face` (LivePortrait). Combine with Phase 1/2 output for "full clone" mode. |
| 4 | Style transfer: `/analyze-style` + Remotion re-edit driver. |
| 5 | UI + job queue: Next.js app, SQLite job store, job runner wiring all stages, consent-gate enforcement. |
| 6 | Assembly polish: captions/branding via Remotion, final export settings. |
| 7 | Hardening: input validation, VRAM serialization, failure recovery, manual test-clip suite. |

Each phase is runnable/demoable standalone via CLI before Phase 5 wires it into the UI, so working pieces exist early instead of one big-bang integration.

## 9. Tech Stack Summary

- **ML inference**: Python, FastAPI, PyTorch, XTTS-v2, LivePortrait, MuseTalk, OpenCV, PySceneDetect.
- **UI**: Next.js (local-only, no deploy), matching the existing `apps/web` stack.
- **Job orchestration**: SQLite + a Node job runner, following `vidgen`'s poller pattern but fully local.
- **Assembly**: Remotion, reused from `vidgen`.
- **Hardware target**: local NVIDIA GPU, 8GB+ VRAM.
- **No cloud APIs** — consistent with `vidgen`'s local-only philosophy (local Ollama + local Kokoro).

## 10. Explicitly Out of Scope

- Cloning the face/voice of any person without their explicit, recorded consent (including public figures/influencers found online).
- Any "paste a URL, clone whoever's in it" convenience feature.
- Cloud API integration (at least for v1 — local-only by design).
