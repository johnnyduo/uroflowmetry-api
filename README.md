# Uroflowmetry API

FastAPI backend for processing audio-based uroflowmetry data.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run locally: `uvicorn main:app --reload`
3. API docs available at `/docs`

## Endpoints
- POST /predict/ - Process audio file
- GET /health - Health check