# Full-Clone Video Platform — Design

**Status:** approved, ready for implementation planning
**Scope:** v1 of a private, team-only web platform for generating full face+voice clone videos. Voice-only clone and style-transfer modes are explicitly out of scope for this spec.

## 1. Goal

Let a small trusted team (not the public) generate a video by supplying:
- a **reference video** (face + voice to clone),
- optionally, their **own footage** (used as b-roll),
- a **brief** describing what the video should say.

The platform clones the reference speaker's face and voice, writes a script from the brief, and produces a final video of the reference person speaking that script — optionally cut with the user's own footage as supporting visuals.

This extends the existing `refclone` project (currently a FastAPI skeleton with only a `/health` endpoint) and reuses `avatar-gen/SadTalker`, which is already vendored in this workspace.

## 2. Non-goals for v1

- No public signup / multi-tenant SaaS concerns (auth, billing, abuse prevention). This is for the owner and a small team, run on one local GPU box.
- No voice-only clone or style-transfer modes — full clone (audio-driven) only.
- No motion-transfer from the user's own footage onto the cloned face — the face's performance is driven by the synthesized audio (SadTalker), not by footage motion.
- No remote/multi-machine deployment — web app, job runner, and GPU inference all run on one machine.
- No job retries or prioritization — one job at a time, straight through; failures require manual re-submission.
- No automated UI tests for the Next.js app in v1 — manual browser verification only.

## 3. Architecture

Three services on one machine:

```
┌─────────────────┐  create job   ┌──────────────────┐  poll/update   ┌─────────────────────┐
│  Next.js web app │──────────────▶│   jobs.sqlite     │◀───────────────│  Node job runner     │
│  (refclone/web)  │◀──────────────│  (jobs, consent,  │────────────────▶│  (refclone/runner)   │
│                  │  poll status  │   stage results)  │                 │                       │
└─────────────────┘                └──────────────────┘                └──────────┬───────────┘
                                                                                    │ HTTP, one stage
                                                                                    │ at a time
                                                                                    ▼
                                                                     ┌───────────────────────────┐
                                                                     │  refclone FastAPI server    │
                                                                     │  (stateless per call)       │
                                                                     └───────────────────────────┘
```

- **Next.js never calls FastAPI directly.** It only reads/writes `jobs.sqlite`.
- **The Node runner is the only caller of FastAPI**, and processes exactly one job at a time (single GPU — no benefit to concurrency).
- **FastAPI stays stateless**: every endpoint takes explicit inputs and returns outputs. It has no concept of "jobs" — that's the runner's job.

## 4. Components

### 4.1 `refclone/` (existing repo — FastAPI, extended)

New modules added to the existing skeleton (`src/refclone/main.py`, plus the Phase 0-2 plan's `gpu.py`, `audio_utils.py`, `voice/enroll.py`, `voice/synthesize.py`):

- `src/refclone/face/sadtalker.py` — subprocess wrapper around `avatar-gen/SadTalker`. Takes a face image + an audio file, returns a talking-head video path. Mirrors the existing `lipsync/musetalk.py` wrapper pattern from the Phase 0-2 plan (subprocess invocation of a vendored model).
- `src/refclone/llm/script_writer.py` — brief + target duration → script text, via local Ollama. Same approach vidgen already uses for its content-calendar scripts.
- New endpoints on the FastAPI app: `/extract-reference` (pull a clean face frame + voice sample from the reference video), `/clone-voice` (existing plan), `/write-script` (new), `/synthesize` (existing plan), `/generate-face` (new, wraps `sadtalker.py`).

### 4.2 `refclone/runner/` (new — Node/TS)

- Owns `jobs.sqlite`: schema for `jobs` (id, status, stage, error, output_path, timestamps), `consent_records` (subject, rights note, timestamp, reference-video file hash), `stage_results` (per-job intermediate artifact paths).
- Polls for `queued` jobs. On claim, checks the job has a linked `consent_records` row — refuses to proceed otherwise (mirrors the CLI's `--i-have-consent` flag, but DB-backed, matching what the original plan describes as the eventual real consent gate).
- Calls FastAPI stage by stage in sequence (listed in §5), updating `jobs.stage` after each so the web app can show progress.
- On any stage failure: marks the job `failed`, records the stage name and error message. No retry logic.
- On success: marks `complete`, records the final output path.

### 4.3 `refclone/web/` (new — Next.js)

- Submission form: reference video upload, optional footage upload(s), brief text field, consent attestation (subject = self / licensed third party + confirmation checkbox).
- On submit: writes a `queued` row + linked `consent_records` row directly to `jobs.sqlite`. No FastAPI call from here.
- Job page: polls the job's row for `status`/`stage`, shows a progress indicator, and once `complete`, shows a preview player + download link for the output.
- No ML logic, no direct FastAPI calls — purely a UI over the SQLite job store.

### 4.4 Assembly (captions + optional b-roll)

A small, brand-agnostic Remotion composition — separate from vidgen's Inferr-branded templates, since this platform generates arbitrary user content, not Inferr marketing videos. Responsibilities: burn in captions from the synthesized speech, crop/pad to a **default 9:16 vertical aspect ratio** (matching SadTalker's natural output framing and vidgen's convention; not user-configurable in v1), and, if the user supplied footage, cut it in as b-roll around the talking-head clip. Invoked as the final pipeline stage.

## 5. Pipeline (one job)

1. **Submit** — user provides reference video, optional footage, brief, consent attestation → Next.js creates a `queued` job + consent record in `jobs.sqlite`.
2. **Claim** — runner picks up the oldest `queued` job with a valid consent record; refuses (leaves `queued`, logs a warning) if consent is missing.
3. **Extract reference** — FastAPI pulls a clean face frame and a clean voice sample from the reference video.
4. **Clone voice** — FastAPI enrolls the voice sample (XTTS-v2), returns a `voice_id`.
5. **Write script** — FastAPI turns the brief (+ target duration) into script text via local Ollama.
6. **Synthesize** — FastAPI synthesizes the script in the cloned voice (XTTS-v2), returns narration audio.
7. **Generate face** — FastAPI runs SadTalker with the reference face frame + narration audio, returns a talking-head video.
8. **Assemble** — Remotion burns in captions from the narration, crops to target aspect ratio, optionally cuts in user footage as b-roll, exports the final MP4.
9. **Complete** — runner records the output path and marks the job `complete`. Any failure at any stage marks the job `failed` with the stage name + error, and stops (no retry).

## 6. Consent enforcement

Every job must have a linked `consent_records` row before the runner will claim it: subject label (`self` or `licensed third party` with a rights note), a confirmation flag, a timestamp, and a hash of the reference video file. This is enforced in the runner (server-side), not just as a frontend checkbox — matching the hard constraint from the original `planforreferencevideo.md` plan. It replaces the interim CLI `--i-have-consent` flag from the Phase 0-2 plan with the real DB-backed gate that plan always intended to arrive later.

## 7. Error handling

- Each pipeline stage is a single FastAPI call; the runner treats any non-2xx response or thrown error as a stage failure.
- On failure: job status → `failed`, `stage` and `error` columns record where and why. No automatic retry — the user re-submits a new job.
- The web app surfaces `failed` jobs with the stage and error message so the user knows what to fix (e.g., unusable reference audio) before retrying.

## 8. Testing

- **`refclone`**: pytest per module, following the existing pattern from the Phase 0-2 plan (`test_gpu.py`, `test_audio_utils.py`, `test_enroll.py`, `test_synthesize.py`), plus new `test_sadtalker.py` and `test_script_writer.py`, and `test_api.py` coverage for the new endpoints.
- **`refclone/runner`**: tests for job state transitions (`queued` → `running` → `complete`/`failed`) and for the consent-gate refusal path.
- **`refclone/web`**: manual verification in-browser for v1 — submit a job, watch it progress, confirm the final video plays and downloads. No automated UI tests yet (small internal tool, not worth the investment at this stage).

## 9. Open questions deferred to later phases

- Voice-only clone and style-transfer modes (same runner/web/FastAPI shape, different pipeline stages).
- Any path toward exposing this beyond the current private team (auth, stricter identity verification, abuse prevention) — explicitly not designed here.
- Retry/queue-priority behavior, if job volume ever grows.
