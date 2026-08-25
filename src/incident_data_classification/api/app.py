from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..baseline_scoring import get_baseline_classes, get_baseline_scores, load_baseline_pipeline
from ..confidence import softmax
from ..config import (
    DEFAULT_MODELS_DIR,
    FEATURE_PROFILE_ALERT_ONLY,
    FEATURE_PROFILES,
)
from ..data import normalize_text
from ..explainability import explain_baseline_prediction
from ..predict_baseline import load_calibration
from ..retrieval import RetrievalIndex, load_retrieval_index, retrieve_similar_incidents


LOGGER = logging.getLogger("incident_data_classification.api")
LINEAR_SVM_MODEL_NAME = "linear_svm"
DEFAULT_API_FEATURE_PROFILE = FEATURE_PROFILE_ALERT_ONLY


@dataclass(frozen=True)
class ApiSettings:
    models_dir: Path = DEFAULT_MODELS_DIR
    feature_profile: str = DEFAULT_API_FEATURE_PROFILE
    top_k: int = 3
    top_features: int = 8
    default_confidence_threshold: float = 0.75

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            models_dir=Path(os.getenv("INCIDENT_MODELS_DIR", str(DEFAULT_MODELS_DIR))),
            feature_profile=os.getenv("INCIDENT_FEATURE_PROFILE", DEFAULT_API_FEATURE_PROFILE),
            top_k=int(os.getenv("INCIDENT_TOP_K", "3")),
            top_features=int(os.getenv("INCIDENT_TOP_FEATURES", "8")),
            default_confidence_threshold=float(os.getenv("INCIDENT_CONFIDENCE_THRESHOLD", "0.75")),
        )


class IncidentClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    severity: str | None = None
    affected_services: list[str] = Field(default_factory=list)
    primary_affected_service: str | None = None
    anomalies: list[str] = Field(default_factory=list)
    environment: str | None = None
    cloud_provider: str | None = None
    region: str | None = None
    timeline_summary: str | None = None
    confidence_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=10)
    top_features: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def require_text_or_alert_fields(self) -> "IncidentClassificationRequest":
        if self.text and self.text.strip():
            return self
        has_alert_fields = bool(
            (self.title and self.title.strip())
            or self.affected_services
            or self.primary_affected_service
            or self.anomalies
        )
        if not has_alert_fields:
            raise ValueError("Provide text or at least one alert field")
        return self


class AlternativeClassification(BaseModel):
    classification: str
    confidence: float


class SupportingSignal(BaseModel):
    term: str
    tfidf_value: float
    weight: float
    contribution: float


class Explanation(BaseModel):
    method: str
    classification: str
    supporting_signals: list[SupportingSignal]
    important_features: list[str]
    note: str


class Evidence(BaseModel):
    model_supporting_signals: list[str]
    retrieved_incident_ids: list[str]
    note: str


class SimilarIncident(BaseModel):
    incident_id: str
    root_cause_category: str
    similarity: float
    title: str
    affected_services: str
    anomaly_types_detected: str
    remediation: str
    prevention_recommendation: str


class ClassificationResponse(BaseModel):
    request_id: str
    model: str
    model_version: str
    feature_profile: str
    classification: str
    confidence: float
    requires_human_review: bool
    confidence_threshold: float
    threshold_source: str
    temperature: float
    inference_latency_ms: float
    alternatives: list[AlternativeClassification]
    explanation: Explanation
    evidence: Evidence
    similar_incidents: list[SimilarIncident]
    recommended_remedy: str | None
    remedy_source: str | None
    remedy_source_similarity: float | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    retrieval_loaded: bool
    error: str | None = None


class ModelMetadataResponse(BaseModel):
    model: str
    model_version: str
    feature_profile: str
    classes: list[str]
    calibration_loaded: bool
    retrieval_loaded: bool
    supports_abstention: bool
    supports_explanations: bool
    supports_similar_incidents: bool


class LinearSVMIncidentService:
    def __init__(self, settings: ApiSettings) -> None:
        if settings.feature_profile not in FEATURE_PROFILES:
            supported = ", ".join(FEATURE_PROFILES)
            raise ValueError(f"Unsupported feature profile {settings.feature_profile!r}. Supported profiles: {supported}")

        self.settings = settings
        self.pipeline = load_baseline_pipeline(settings.models_dir, LINEAR_SVM_MODEL_NAME, settings.feature_profile)
        self.classes = get_baseline_classes(self.pipeline)
        self.artifact_dir = settings.models_dir / "baselines" / LINEAR_SVM_MODEL_NAME / settings.feature_profile
        self.calibration = load_calibration(self.artifact_dir)
        self.retrieval_index = self._load_retrieval_index()
        self.model_version = self._build_model_version()

    def _load_retrieval_index(self) -> RetrievalIndex | None:
        retrieval_path = self.settings.models_dir / "retrieval" / self.settings.feature_profile / "index.joblib"
        if not retrieval_path.exists():
            return None
        return load_retrieval_index(retrieval_path)

    def _build_model_version(self) -> str:
        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            return f"{LINEAR_SVM_MODEL_NAME}:{self.settings.feature_profile}:missing"
        return f"{LINEAR_SVM_MODEL_NAME}:{self.settings.feature_profile}:{int(model_path.stat().st_mtime)}"

    def metadata(self) -> ModelMetadataResponse:
        return ModelMetadataResponse(
            model=LINEAR_SVM_MODEL_NAME,
            model_version=self.model_version,
            feature_profile=self.settings.feature_profile,
            classes=self.classes,
            calibration_loaded=self.calibration is not None,
            retrieval_loaded=self.retrieval_index is not None,
            supports_abstention=True,
            supports_explanations=True,
            supports_similar_incidents=self.retrieval_index is not None,
        )

    def classify(self, payload: IncidentClassificationRequest, request_id: str) -> ClassificationResponse:
        start = time.perf_counter()
        text = build_incident_text(payload)
        threshold, threshold_source = self._resolve_threshold(payload.confidence_threshold)
        temperature = float(self.calibration["temperature"]) if self.calibration is not None else 1.0
        scores = get_baseline_scores(self.pipeline, [text])
        probabilities = softmax(scores, temperature=temperature)[0]
        ranked_indices = np.argsort(probabilities)[::-1]
        top_index = int(ranked_indices[0])
        classification = self.classes[top_index]
        confidence = float(probabilities[top_index])
        top_k = payload.top_k or self.settings.top_k
        top_features = payload.top_features or self.settings.top_features

        similar_incidents = []
        if self.retrieval_index is not None:
            similar_incidents = retrieve_similar_incidents(
                self.retrieval_index,
                text,
                category=classification,
                top_k=top_k,
            )

        explanation = explain_baseline_prediction(
            self.pipeline,
            text,
            predicted_label=classification,
            top_n=top_features,
        )
        top_match = similar_incidents[0] if similar_incidents else None
        latency_ms = (time.perf_counter() - start) * 1000

        response = ClassificationResponse(
            request_id=request_id,
            model=LINEAR_SVM_MODEL_NAME,
            model_version=self.model_version,
            feature_profile=self.settings.feature_profile,
            classification=classification,
            confidence=confidence,
            requires_human_review=confidence < threshold,
            confidence_threshold=threshold,
            threshold_source=threshold_source,
            temperature=temperature,
            inference_latency_ms=latency_ms,
            alternatives=[
                AlternativeClassification(
                    classification=self.classes[int(index)],
                    confidence=float(probabilities[int(index)]),
                )
                for index in ranked_indices[1:3]
            ],
            explanation=explanation,
            evidence=Evidence(
                model_supporting_signals=explanation["important_features"],
                retrieved_incident_ids=[str(match["incident_id"]) for match in similar_incidents],
                note=(
                    "Evidence combines model-supporting TF-IDF signals and similar historical incidents. "
                    "It is not causal proof."
                ),
            ),
            similar_incidents=[SimilarIncident(**match) for match in similar_incidents],
            recommended_remedy=str(top_match["remediation"]) if top_match else None,
            remedy_source=str(top_match["incident_id"]) if top_match else None,
            remedy_source_similarity=float(top_match["similarity"]) if top_match else None,
        )
        LOGGER.info(
            "incident_classified",
            extra={
                "request_id": request_id,
                "classification": classification,
                "requires_human_review": response.requires_human_review,
                "inference_latency_ms": latency_ms,
            },
        )
        return response

    def _resolve_threshold(self, override: float | None) -> tuple[float, str]:
        if override is not None:
            return override, "request"
        if self.calibration is not None:
            return float(self.calibration["threshold"]), "calibration"
        return self.settings.default_confidence_threshold, "default"


def build_incident_text(payload: IncidentClassificationRequest) -> str:
    if payload.text and payload.text.strip():
        return normalize_text(payload.text)

    parts = [
        payload.title,
        payload.severity,
        " ".join(payload.affected_services),
        payload.primary_affected_service,
        " ".join(payload.anomalies),
        payload.environment,
        payload.cloud_provider,
        payload.region,
        payload.timeline_summary,
    ]
    return " ".join(normalize_text(part) for part in parts if part)


def get_service(app: FastAPI) -> LinearSVMIncidentService:
    service = getattr(app.state, "service", None)
    if service is None:
        detail = getattr(app.state, "startup_error", None) or "Classifier service is not ready"
        raise HTTPException(status_code=503, detail=detail)
    return service


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.service = LinearSVMIncidentService(resolved_settings)
            app.state.startup_error = None
        except Exception as exc:  # pragma: no cover - exercised through health checks.
            app.state.service = None
            app.state.startup_error = str(exc)
            LOGGER.exception("incident_api_startup_failed")
        yield

    app = FastAPI(
        title="Incident RCA Classifier",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service = getattr(request.app.state, "service", None)
        error = getattr(request.app.state, "startup_error", None)
        return HealthResponse(
            status="ok" if service is not None else "error",
            model_loaded=service is not None,
            retrieval_loaded=bool(service and service.retrieval_index is not None),
            error=error,
        )

    @app.get("/v1/model", response_model=ModelMetadataResponse)
    def model_metadata(request: Request) -> ModelMetadataResponse:
        return get_service(request.app).metadata()

    @app.post("/v1/incidents/classify", response_model=ClassificationResponse)
    def classify_incident(
        payload: IncidentClassificationRequest,
        request: Request,
        x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
    ) -> ClassificationResponse:
        request_id = x_request_id or str(uuid.uuid4())
        return get_service(request.app).classify(payload, request_id=request_id)

    return app


app = create_app()
