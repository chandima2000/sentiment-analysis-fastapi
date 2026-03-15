"""
app/main.py — FastAPI Application Entry Point
==============================================

Endpoints:
  GET  /health          → liveness check
  POST /predict         → single text sentiment prediction
  POST /predict/batch   → batch sentiment prediction (bonus)

Design decisions:
  - lifespan context manager (asynccontextmanager) is used for startup/shutdown
    logic instead of the deprecated @app.on_event("startup") decorator. This
    is the recommended pattern in FastAPI 0.93+ and avoids deprecation warnings.
  - The model is loaded once at startup and stored in app/model.py's module
    scope. This means every request hits an already-loaded in-memory pipeline
    rather than reading from disk, keeping per-request latency minimal.
  - All endpoints use typed Pydantic schemas for both input and output —
    no raw dicts. This ensures full OpenAPI documentation is auto-generated
    and all data is validated before reaching business logic.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import load_model, predict, predict_batch
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.

    Why lifespan instead of @app.on_event?
    The on_event decorator was deprecated in FastAPI 0.93. The lifespan
    context manager is the current best practice — it groups startup and
    shutdown logic in one place and works correctly with async test clients.

    Model loading here (rather than lazily on first request) means:
      1. The first real request is just as fast as any subsequent one.
      2. If the model file is missing, the server fails immediately at boot
         rather than returning a 500 error to the first unlucky caller.
    """
    # --- Startup ---
    load_model()

    yield  # Application runs while suspended here

    # --- Shutdown --- (nothing to clean up for a pickle/joblib model)


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "A REST API that classifies text sentiment as **positive** or **negative** "
        "using a TF-IDF + Logistic Regression model trained on the IMDB dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness check",
    description="Returns `{'status': 'ok'}` when the service is running and ready.",
)
def health_check() -> HealthResponse:
    """
    Simple liveness endpoint.

    Used by infrastructure tooling (load balancers, container orchestrators)
    to verify the service is up. Returning a static response is intentional —
    a health check that calls the model adds latency and failure modes.
    """
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment for a single text",
    description=(
        "Accepts a JSON body `{'text': '...'}` and returns the predicted sentiment "
        "(`positive` or `negative`) along with a confidence score (0.0–1.0)."
    ),
)
def predict_sentiment(request: PredictRequest) -> PredictResponse:
    """
    Run sentiment classification on a single text input.

    The text is passed directly to the trained scikit-learn pipeline, which
    applies TF-IDF vectorization and Logistic Regression in one step.
    Confidence is the predicted class probability from predict_proba —
    meaningful because Logistic Regression is a well-calibrated classifier.
    """
    try:
        result = predict(request.text)
    except RuntimeError as exc:
        # This should not occur in normal operation because load_model() is
        # called at startup. Exposed as 503 to signal a temporary server issue.
        raise HTTPException(status_code=503, detail=str(exc))

    return PredictResponse(**result)


# ---------------------------------------------------------------------------
# POST /predict/batch 
# ---------------------------------------------------------------------------

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Predict sentiment for a list of texts",
    description=(
        "Accepts `{'texts': ['...', '...']}` and returns predictions for all items "
        "in a single request. More efficient than calling `/predict` in a loop because "
        "TF-IDF vectorization and matrix operations run over the whole batch at once. "
        "Returns HTTP 422 for an empty `texts` list."
    ),
)
def predict_batch_sentiment(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Run sentiment classification on a batch of texts.

    Batching is significantly faster than multiple single-text requests:
    the TF-IDF sparse matrix for all inputs is built in one pass, and
    Logistic Regression scoring is a single matrix multiplication over
    the entire batch rather than n separate vector dot products.
    """
    try:
        results = predict_batch(request.texts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    predictions = [PredictResponse(**r) for r in results]
    return BatchPredictResponse(predictions=predictions)
