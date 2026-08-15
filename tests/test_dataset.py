from pathlib import Path

from incident_data_classification.config import DEFAULT_CSV_PATH
from incident_data_classification.dataset import resolve_incidents_csv


def test_default_dataset_path_uses_local_raw_folder():
    parts = DEFAULT_CSV_PATH.parts

    assert "data" in parts
    assert "raw" in parts
    assert DEFAULT_CSV_PATH.name == "incidents_training_dataset.csv"


def test_resolve_incidents_csv_honors_explicit_local_path():
    path = Path("custom/incidents_training_dataset.csv")

    assert resolve_incidents_csv(path) == path

