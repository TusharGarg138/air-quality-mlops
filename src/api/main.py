from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.schemas import AirQualityInput, PredictionOutput, HealthResponse
from src.inference.predict import AQIPredictor

predictor: AQIPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = AQIPredictor()
        print("Model and preprocessor artifacts loaded successfully into memory.")
    except Exception as e:
        print(f"Warning on startup artifact load: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Air Quality Index Prediction API",
    description="CPCB-compliant production AQI estimation service built with FastAPI & Docker.",
    version="1.0.0",
    lifespan=lifespan
)

# Serve web frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_home():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "Air Quality API is running. Access /docs for swagger specifications."}


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    is_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded
    )


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict_aqi(payload: AirQualityInput):
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not ready. Model artifacts are missing or uninitialized."
        )
    try:
        result = predictor.predict(payload.model_dump())
        return result
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(err)}"
        )