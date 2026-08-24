from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_FEATURE_PROFILE, TARGET_COLUMN
from .data import normalize_text


RETRIEVAL_METADATA_COLUMNS = [
    "incident_id",
    "root_cause_category",
    "remediation_that_worked",
    "prevention_recommendation",
    "title",
    "affected_services",
    "primary_affected_service",
    "anomaly_types_detected",
]


@dataclass
class RetrievalIndex:
    feature_profile: str
    vectorizer: TfidfVectorizer
    matrix: Any
    records: list[dict[str, str]]


def build_retrieval_index(
    incidents: pd.DataFrame,
    feature_profile: str = DEFAULT_FEATURE_PROFILE,
) -> RetrievalIndex:
    if "input_text" not in incidents.columns:
        raise ValueError("incidents must include an input_text column")
    if TARGET_COLUMN not in incidents.columns:
        raise ValueError(f"incidents must include {TARGET_COLUMN!r}")

    texts = incidents["input_text"].fillna("").map(normalize_text).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts).tocsr()
    records = build_records(incidents)
    return RetrievalIndex(
        feature_profile=feature_profile,
        vectorizer=vectorizer,
        matrix=matrix,
        records=records,
    )


def build_records(incidents: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for _, row in incidents.reset_index(drop=True).iterrows():
        record = {}
        for column in RETRIEVAL_METADATA_COLUMNS:
            value = row.get(column, "")
            record[column] = "" if pd.isna(value) else str(value)
        record["input_text"] = normalize_text(row.get("input_text", ""))
        records.append(record)
    return records


def save_retrieval_index(path: Path, index: RetrievalIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_profile": index.feature_profile,
            "vectorizer": index.vectorizer,
            "matrix": index.matrix,
            "records": index.records,
        },
        path,
    )


def load_retrieval_index(path: Path) -> RetrievalIndex:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval index not found at {path}. Build it first.")
    payload = joblib.load(path)
    return RetrievalIndex(
        feature_profile=payload["feature_profile"],
        vectorizer=payload["vectorizer"],
        matrix=payload["matrix"].tocsr(),
        records=payload["records"],
    )


def retrieve_similar_incidents(
    index: RetrievalIndex,
    query_text: str,
    *,
    category: str | None = None,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    candidate_indices = [
        record_index
        for record_index, record in enumerate(index.records)
        if category is None or record["root_cause_category"] == category
    ]
    if not candidate_indices:
        return []

    query_vector = index.vectorizer.transform([normalize_text(query_text)])
    candidate_matrix = index.matrix[candidate_indices]
    similarities = cosine_similarity(query_vector, candidate_matrix)[0]
    ranked_positions = similarities.argsort()[::-1][:top_k]

    matches: list[dict[str, str | float]] = []
    for position in ranked_positions:
        record_index = candidate_indices[int(position)]
        record = index.records[record_index]
        matches.append(
            {
                "incident_id": record["incident_id"],
                "root_cause_category": record["root_cause_category"],
                "similarity": float(similarities[int(position)]),
                "title": record["title"],
                "affected_services": record["affected_services"],
                "anomaly_types_detected": record["anomaly_types_detected"],
                "remediation": record["remediation_that_worked"],
                "prevention_recommendation": record["prevention_recommendation"],
            }
        )
    return matches
