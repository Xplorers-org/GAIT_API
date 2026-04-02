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
    - `patient_gender`: `male` or `female`

Response includes:

- clinical interpretation text (same wording/thresholds as notebook)
- gait score and interpretation
- full feature values
- URL to annotated output video
- URL to biomarker plot image
