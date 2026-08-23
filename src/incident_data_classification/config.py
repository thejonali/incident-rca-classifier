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

FEATURE_PROFILE_ALERT_ONLY = "alert_only"
FEATURE_PROFILE_EARLY_INCIDENT = "early_incident"
FEATURE_PROFILE_POSTMORTEM = "postmortem"
DEFAULT_FEATURE_PROFILE = FEATURE_PROFILE_EARLY_INCIDENT

FEATURE_PROFILE_COLUMNS = {
    FEATURE_PROFILE_ALERT_ONLY: [
        "title",
        "severity",
        "affected_services",
        "primary_affected_service",
        "anomaly_types_detected",
    ],
    FEATURE_PROFILE_EARLY_INCIDENT: [
        "title",
        "severity",
        "affected_services",
        "primary_affected_service",
        "anomaly_types_detected",
        "environment",
        "cloud_provider",
        "region",
    ],
    FEATURE_PROFILE_POSTMORTEM: [
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
    ],
}

FEATURE_PROFILES = tuple(FEATURE_PROFILE_COLUMNS)

# Backward-compatible alias for callers that expect a single configured input.
INPUT_COLUMNS = FEATURE_PROFILE_COLUMNS[DEFAULT_FEATURE_PROFILE]

INCIDENT_TIME_EXCLUDED_COLUMNS = {
    "root_cause_description",
    "contributing_factors",
    "timeline_summary",
    "remediation_steps_taken",
    "remediation_that_worked",
    "post_mortem_summary",
    "prevention_recommendation",
}

TARGET_COLUMN = "root_cause_category"

LEAKY_COLUMNS = {
    "root_cause_category",
    "root_cause_description",
    "contributing_factors",
    "timeline_summary",
    "remediation_steps_taken",
    "remediation_that_worked",
    "post_mortem_summary",
    "prevention_recommendation",
}
