---
title: GAIT_API
emoji: 🚶
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Gait Analysis API

Clinical gait analysis service built with FastAPI, OpenCV, and MediaPipe Pose.

It processes a front-view walking video and returns:

- extracted gait biomarkers
- rule-based clinical interpretation
- overall gait stability score
- annotated skeleton video
- clinical dashboard plot

The notebook prototype is kept in [gait.ipynb](<gait.ipynb>), and the production API implementation is in [app.py](app.py).

## Table of contents

- Overview
- Project structure
- How it works
- API reference
- Local development
- Docker usage
- Storage cleanup strategy
- Hugging Face Spaces deployment
- GitHub Actions auto-deploy
- Troubleshooting

## Overview

This API is designed for single-video gait assessment.

Core stack:

- FastAPI for REST endpoints
- MediaPipe Pose for landmark extraction
- OpenCV for video I/O and skeleton overlay
- NumPy/SciPy for signal processing and feature extraction
- Matplotlib for biomarker visualizations

Dependencies are listed in [requirements.txt](requirements.txt).

## Project structure

- [app.py](app.py): Main API + gait analysis pipeline
- [requirements.txt](requirements.txt): Python dependencies
- [Dockerfile](Dockerfile): Container build (HF Spaces compatible)
- [scripts/start.sh](scripts/start.sh): Container startup + background cleanup loop
- [scripts/cleanup_runs.py](scripts/cleanup_runs.py): Deletes old generated files
- [.github/workflows/deploy-hf-space.yml](.github/workflows/deploy-hf-space.yml): Auto-sync GitHub repo to HF Space
- [gait.ipynb](<gait.ipynb>): Original notebook source logic

## How it works

High-level flow:

1. Upload `video` + `gender`
2. Extract pose landmarks for each frame
3. Validate video (person detected, front-view check)
4. Build temporal signals (ankles, feet, arm swing, hip center)
5. Smooth + detrend + detect peaks
6. Compute biomarkers (`stride_variability`, `cadence`, `symmetry_ratio`, arm metrics)
7. Create clinical interpretation text
8. Compute weighted gait stability score
9. Generate dashboard image + annotated video
10. Return JSON payload

Main endpoints are declared in [app.py](app.py#L560-L785).

## API reference

### `GET /`

Basic API metadata and endpoint hints.

### `POST /analyze`

Accepts multipart form-data:

- `video`: gait video (`mp4/mov/avi/...`)
- `gender`: `male` or `female`

Returns analysis JSON with base64-embedded files (`annotated_video`, `clinical_dashboard`).

Use this when you want everything in one response.

### `POST /analyze_files`

Accepts multipart form-data:

- `video`: gait video
- `gender`: `male` or `female`

Returns analysis JSON with downloadable URLs:

- `/download/{session_id}_annotated.mp4`
- `/download/{session_id}_dashboard.png`

This is generally the better choice for deployment because responses stay smaller than full base64 payloads.

### `GET /download/{filename}`

Downloads generated output files from `runs/outputs`.

### `GET /health`

Simple health check.


## Local development

1. Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Open docs

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Docker usage

Build:

```bash
docker build -t gait-api:latest .
```

Run:

```bash
docker run --rm -p 7860:7860 gait-api:latest
```

Container defaults:

- serves on port `7860`
- startup script: [scripts/start.sh](scripts/start.sh)
- output directory: `/app/runs/outputs`

## Storage cleanup strategy

Generated files from `/analyze_files` are stored under `runs/outputs`.

Cleanup is handled by [scripts/cleanup_runs.py](scripts/cleanup_runs.py):

- default retention: 30 minutes
- deletes old files under `runs/`
- preserves required directory structure

In Docker/HF Spaces, [scripts/start.sh](scripts/start.sh) starts a background cleanup loop automatically.

Configurable environment variables:

- `CLEANUP_INTERVAL_SECONDS` (default: `1800`)
- `RUNS_MAX_AGE_MINUTES` (default: `30`)

Optional manual run:

```bash
python scripts/cleanup_runs.py --path ./runs --max-age-minutes 30 --dry-run
```

## Hugging Face Spaces deployment (Docker)

This repository is configured for Docker Spaces.

Key points:

- README front matter is required and already included
- container uses [Dockerfile](Dockerfile)
- app starts via [scripts/start.sh](scripts/start.sh)
- `PORT` env is respected (default `7860`)

Recommended endpoint on Spaces:

- Use `/analyze_files` for better response size and reliability

## GitHub Actions auto-deploy to HF Space

Workflow: [.github/workflows/deploy-hf-space.yml](.github/workflows/deploy-hf-space.yml)

Behavior:

- triggers on push to `main`
- sanitizes `HF_TOKEN`
- force-pushes repository to `xplorers/GAIT_API`

Required GitHub secret:

- `HF_TOKEN`: Hugging Face token with write access to the target Space
