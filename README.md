# HEMAS NeuroTrack FastAPI

This project ports the complete notebook logic from `gait2 (5).ipynb` into a FastAPI app with Swagger UI.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open Swagger UI:

- http://127.0.0.1:8000/docs

## Endpoint

- `POST /analyze`
  - form-data:
    - `video`: video file (front-view gait)
    - `gender`: `male` or `female`

- `POST /analyze_files`
  - form-data:
    - `video`: video file
    - `gender`: `male` or `female`

- `GET /download/{filename}`
- `GET /health`

Response includes:

- clinical interpretation text (same wording/thresholds as notebook)
- gait score and interpretation
- full feature values
- URL to annotated output video
- URL to biomarker plot image

## Automatic cleanup for runs/

Generated files from `/analyze_files` are stored under `runs/outputs`.
To prevent storage growth, use the cleanup script every 30 minutes.

Script path:

- `scripts/cleanup_runs.py`

Behavior:

- Deletes files older than 30 minutes (default)
- Removes empty subdirectories

### Local host cron example

```bash
*/30 * * * * /usr/bin/python3 /path/to/GAIT_API/scripts/cleanup_runs.py --path /path/to/GAIT_API/runs --max-age-minutes 30 >> /var/log/gait_cleanup.log 2>&1
```

### Docker container cron example (host cron running docker exec)

```bash
*/30 * * * * docker exec gait-api python /app/scripts/cleanup_runs.py --path /app/runs --max-age-minutes 30 >> /var/log/gait_cleanup.log 2>&1
```

Replace `gait-api` with your running container name.

## Hugging Face Spaces (Docker) deployment

This repository is ready for Docker Spaces deployment. The container now:

- listens on `PORT` (default `7860`) required by Spaces
- starts a background cleanup loop for `/app/runs`
- launches the API via `scripts/start.sh`

Environment variables (optional):

- `PORT` (default: `7860`)
- `CLEANUP_INTERVAL_SECONDS` (default: `1800`)
- `RUNS_MAX_AGE_MINUTES` (default: `30`)

## GitHub Actions auto-deploy to your HF Space

Workflow file:

- `.github/workflows/deploy-hf-space.yml`

Target Space:

- `xplorers/GAIT_API`

### Required GitHub secret

Add this repository secret in GitHub:

- `HF_TOKEN` = a Hugging Face User Access Token with write permission to `xplorers/GAIT_API`

On every push to `main`, GitHub Actions force-pushes this repo to your Space.
