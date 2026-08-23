# Model Scores

These results come from full-dataset training runs on `incidents_training_dataset.csv`. The dataset is synthetic, so treat these as reproducibility metrics for this experiment rather than claims about real incident RCA performance.

Run configuration:

- Rows used: 15,000
- Test rows: 2,250
- Split strategy: stratified train/validation/test split
- Device: CPU
- Saved artifacts: `models/<model>/<feature_profile>/`

## Feature Profile Matrix

| Model | Feature Profile | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Train Time | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GRU | alert_only | 0.284 | 0.037 | 0.126 | 4 | 54.9s | Balanced class weights |
| GRU | early_incident | 0.284 | 0.037 | 0.126 | 1 | 39.2s | Balanced class weights |
| GRU | postmortem | 0.996 | 0.993 | 0.996 | 18 | 124.6s | Balanced class weights |
| LSTM | alert_only | 0.284 | 0.037 | 0.126 | 1 | 22.6s | Existing tuned LSTM settings |
| LSTM | early_incident | 0.284 | 0.037 | 0.126 | 1 | 24.1s | Existing tuned LSTM settings |
| LSTM | postmortem | 0.922 | 0.721 | 0.884 | 1 | 23.1s | Existing tuned LSTM settings |

The incident-time profiles collapse to the majority-class baseline. `RESOURCE_EXHAUSTION` accounts for 640 of 2,250 test rows, so always predicting that class yields 0.284 accuracy and very low macro F1. Adding environment, cloud provider, and region in `early_incident` did not improve either neural model on this synthetic dataset.

The postmortem profile is dramatically easier because it includes `timeline_summary`, `root_cause_description`, and `contributing_factors`. Those fields are often discovered during or after investigation, so the high postmortem score should not be treated as live-triage performance.

## Postmortem Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Train Time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GRU | 0.996 | 0.993 | 0.996 | 18 | 124.6s | Balanced class weights, 20 epochs |
| LSTM | 0.922 | 0.721 | 0.884 | 1 | 23.1s | Smaller LSTM tuned for accuracy |

The GRU is the stronger postmortem classifier on this dataset. It clears 90% on accuracy, macro F1, and weighted F1. The LSTM clears 90% accuracy, but macro F1 remains weak because it still misses several smaller classes.

## GRU Per-Class Scores

| Classification | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| CASCADING_FAILURE | 1.00 | 1.00 | 1.00 | 98 |
| CONFIGURATION_ERROR | 1.00 | 1.00 | 1.00 | 99 |
| DATA_CORRUPTION | 1.00 | 0.96 | 0.98 | 57 |
| DEPENDENCY_FAILURE | 1.00 | 1.00 | 1.00 | 418 |
| DEPLOYMENT_REGRESSION | 0.99 | 1.00 | 1.00 | 203 |
| INFRASTRUCTURE_FAILURE | 1.00 | 1.00 | 1.00 | 205 |
| RESOURCE_EXHAUSTION | 0.99 | 1.00 | 0.99 | 640 |
| RESOURCE_LEAK | 1.00 | 1.00 | 1.00 | 214 |
| SCHEDULED_JOB_FAILURE | 0.98 | 0.92 | 0.95 | 60 |
| SECURITY_INCIDENT | 1.00 | 1.00 | 1.00 | 100 |
| THIRD_PARTY_FAILURE | 1.00 | 0.98 | 0.99 | 59 |
| TRAFFIC_OVERLOAD | 1.00 | 1.00 | 1.00 | 97 |

## LSTM Per-Class Scores

| Classification | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| CASCADING_FAILURE | 0.91 | 1.00 | 0.95 | 98 |
| CONFIGURATION_ERROR | 0.89 | 1.00 | 0.94 | 99 |
| DATA_CORRUPTION | 0.00 | 0.00 | 0.00 | 57 |
| DEPENDENCY_FAILURE | 0.93 | 1.00 | 0.97 | 418 |
| DEPLOYMENT_REGRESSION | 0.92 | 1.00 | 0.96 | 203 |
| INFRASTRUCTURE_FAILURE | 0.93 | 1.00 | 0.96 | 205 |
| RESOURCE_EXHAUSTION | 0.90 | 1.00 | 0.95 | 640 |
| RESOURCE_LEAK | 0.96 | 1.00 | 0.98 | 214 |
| SCHEDULED_JOB_FAILURE | 0.00 | 0.00 | 0.00 | 60 |
| SECURITY_INCIDENT | 0.95 | 1.00 | 0.98 | 100 |
| THIRD_PARTY_FAILURE | 0.00 | 0.00 | 0.00 | 59 |
| TRAFFIC_OVERLOAD | 0.94 | 1.00 | 0.97 | 97 |

## Current Read

The class-weighted GRU is the strongest classifier in this repository when post-investigation fields are available. It fixed the previously missed minority categories and performs above 95% F1 for every class in this synthetic test split.

The incident-time profiles are the more realistic live-triage benchmarks, and they currently show that the available alert metadata is not enough to infer root cause. This is a dataset limitation, not a neural tuning success case. A targeted class-weighted LSTM postmortem run was also tested and did not improve macro F1.

The LSTM reaches 92.2% accuracy, but that number is inflated by strong performance on the common categories. It still fails to identify:

- `DATA_CORRUPTION`
- `SCHEDULED_JOB_FAILURE`
- `THIRD_PARTY_FAILURE`

## Commands

Recreate the reported GRU profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type gru --feature-profile "$profile" --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
done
```

Recreate the reported LSTM profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type lstm --feature-profile "$profile" --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
done
```

Compare saved results:

```bash
uv run python -m incident_data_classification.evaluate
```
