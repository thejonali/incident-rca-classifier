from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DERIVED_COLUMNS,
    DEFAULT_FEATURE_PROFILE,
    EARLY_TIMELINE_COLUMN,
    EARLY_TIMELINE_EVENT_COUNT,
    FEATURE_PROFILE_POSTMORTEM,
    FEATURE_PROFILE_COLUMNS,
    INCIDENT_TIME_EXCLUDED_COLUMNS,
    INPUT_COLUMNS,
    LEAKY_COLUMNS,
    RANDOM_SEED,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)
from .tokenizer import TextTokenizer


_WHITESPACE_RE = re.compile(r"\s+")
_TIMELINE_EVENT_RE = re.compile(r"[|\r\n]+")


@dataclass(frozen=True)
class DatasetSplits:
    x_train: list[str]
    x_val: list[str]
    x_test: list[str]
    y_train: list[str]
    y_val: list[str]
    y_test: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = text.replace("—", " ")
    text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_early_timeline(value: object, event_count: int = EARLY_TIMELINE_EVENT_COUNT) -> str:
    if event_count <= 0 or pd.isna(value):
        return ""

    events = [event.strip() for event in _TIMELINE_EVENT_RE.split(str(value)) if event.strip()]
    return " ".join(events[:event_count])


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    leaking_inputs = [column for column in INPUT_COLUMNS if column in LEAKY_COLUMNS]
    if leaking_inputs:
        raise ValueError(f"Input column list includes target-leaking columns: {leaking_inputs}")


def get_feature_columns(feature_profile: str = DEFAULT_FEATURE_PROFILE) -> list[str]:
    try:
        columns = FEATURE_PROFILE_COLUMNS[feature_profile]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_PROFILE_COLUMNS))
        raise ValueError(f"Unsupported feature profile {feature_profile!r}. Supported profiles: {supported}") from exc

    if feature_profile != FEATURE_PROFILE_POSTMORTEM:
        excluded_columns = [column for column in columns if column in INCIDENT_TIME_EXCLUDED_COLUMNS]
        if excluded_columns:
            raise ValueError(f"Incident-time profile {feature_profile!r} includes post-investigation columns: {excluded_columns}")

    return list(columns)


def resolve_feature_value(row: pd.Series, column: str) -> object:
    if column == EARLY_TIMELINE_COLUMN:
        return extract_early_timeline(row.get("timeline_summary", ""))
    return row[column]


def build_input_text(row: pd.Series, feature_profile: str = DEFAULT_FEATURE_PROFILE) -> str:
    parts = [normalize_text(resolve_feature_value(row, column)) for column in get_feature_columns(feature_profile)]
    return " ".join(part for part in parts if part)


def load_incidents(
    csv_path: Path,
    max_rows: int | None = None,
    feature_profile: str = DEFAULT_FEATURE_PROFILE,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    for column in get_feature_columns(feature_profile):
        if column in DERIVED_COLUMNS:
            df[column] = df.apply(resolve_feature_value, axis=1, column=column)
    df["input_text"] = df.apply(build_input_text, axis=1, feature_profile=feature_profile)
    df = df[df["input_text"].str.len() > 0].copy()

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df, _ = train_test_split(
            df,
            train_size=max_rows,
            stratify=df[TARGET_COLUMN],
            random_state=RANDOM_SEED,
        )
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15) -> DatasetSplits:
    labels = df[TARGET_COLUMN]
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )

    return DatasetSplits(
        x_train=train_df["input_text"].tolist(),
        x_val=val_df["input_text"].tolist(),
        x_test=test_df["input_text"].tolist(),
        y_train=train_df[TARGET_COLUMN].tolist(),
        y_val=val_df[TARGET_COLUMN].tolist(),
        y_test=test_df[TARGET_COLUMN].tolist(),
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )


def prepare_sequences(
    splits: DatasetSplits,
    vocab_size: int,
    max_length: int,
) -> tuple[TextTokenizer, dict[str, list[list[int]]]]:
    tokenizer = TextTokenizer(vocab_size=vocab_size, max_length=max_length)
    tokenizer.fit(splits.x_train)
    sequences = {
        "train": tokenizer.texts_to_sequences(splits.x_train),
        "val": tokenizer.texts_to_sequences(splits.x_val),
        "test": tokenizer.texts_to_sequences(splits.x_test),
    }
    return tokenizer, sequences
