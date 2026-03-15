"""
app/schemas.py — Pydantic Request & Response Models
====================================================

All API input and output is typed through these models.

Why Pydantic schemas instead of raw dicts?
  - Automatic validation: invalid input is rejected before it reaches business
    logic, with a clear 422 error response and field-level error messages.
  - Self-documenting: FastAPI uses these models to generate the OpenAPI spec
    and Swagger UI automatically — no manual documentation needed.
  - Type safety: editors and linters can catch mismatches at write time.
"""

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for single-text sentiment prediction."""

    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        """
        Reject blank or whitespace-only input before it reaches the model.

        A blank string would not cause a crash in the pipeline, but would
        return a meaningless prediction. Validating here gives the caller a
        clear, actionable error rather than a confusing low-confidence result.
        """
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty or contain only whitespace")
        if len(v) > 10_000:
            raise ValueError("text must not exceed 10,000 characters")
        return v


class BatchPredictRequest(BaseModel):
    """Request body for batch sentiment prediction."""

    texts: list[str]

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: list[str]) -> list[str]:
        """
        Reject empty lists at the schema layer.

        The assignment explicitly requires returning a 400-level error for an
        empty list. Catching it via a Pydantic validator produces a 422
        Unprocessable Entity — the semantically correct HTTP status for a
        well-formed JSON request that fails business validation rules.
        """
        if len(v) == 0:
            raise ValueError("texts list must contain at least one item")
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Response body for a single sentiment prediction."""

    text: str
    # "positive" or "negative" — matching the labels in the training dataset
    sentiment: str
    # Probability of the predicted class (0.0 – 1.0), from predict_proba.
    # Logistic Regression produces well-calibrated probabilities, making
    # this confidence score meaningful and reliable for downstream use.
    confidence: float


class BatchPredictResponse(BaseModel):
    """Response body for batch sentiment predictions."""

    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    """Response body for the liveness health check endpoint."""

    status: str
