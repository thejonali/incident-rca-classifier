from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = "incidents_training_dataset.csv"
DATASET_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sakthivarshans/rca-synthetic-training-dataset"
    "?file_name=incidents_training_dataset.csv"
)
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CSV_PATH = DEFAULT_RAW_DATA_DIR / DATASET_FILE
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "incident_id",
    "severity",
    "title",
    "affected_services",
    "primary_affected_service",
    "anomaly_types_detected",
    "cloud_provider",
    "region",
    "environment",
    "root_cause_category",
    "root_cause_description",
    "contributing_factors",
    "timeline_summary",
    "remediation_that_worked",
    "prevention_recommendation",
]

INPUT_COLUMNS = [
    "title",
    "affected_services",
    "primary_affected_service",
    "anomaly_types_detected",
    "severity",
    "cloud_provider",
    "region",
    "environment",
    "timeline_summary",
    "root_cause_description",
    "contributing_factors",
]

TARGET_COLUMN = "root_cause_category"

LEAKY_COLUMNS = {
    "root_cause_category",
    "remediation_steps_taken",
    "remediation_that_worked",
    "post_mortem_summary",
    "prevention_recommendation",
}
