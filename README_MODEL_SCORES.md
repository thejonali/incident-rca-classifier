# Model Scores

These results come from full-dataset training runs on `incidents_training_dataset.csv`. The dataset is synthetic, so treat these as reproducibility metrics for this experiment rather than claims about real incident RCA performance.

Run configuration:

- Rows used: 15,000
- Test rows: 2,250
- Split strategy: stratified train/validation/test split
- Device: CPU
- Saved artifacts: `models/<model>/<feature_profile>/`

## Feature Profile Matrix

| Model | Feature Profile | Accuracy | Macro F1 | Weighted F1 | Train Time | Inference | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| GRU | alert_only | 0.284 | 0.037 | 0.126 | 54.9s | - | Balanced class weights |
| GRU | early_incident | 0.284 | 0.037 | 0.126 | 35.7s | - | Unweighted loss performed best |
| GRU | postmortem | 0.996 | 0.993 | 0.996 | 124.6s | - | Balanced class weights |
| LSTM | alert_only | 0.284 | 0.037 | 0.126 | 22.6s | - | Existing tuned LSTM settings |
| LSTM | early_incident | 0.284 | 0.037 | 0.126 | 28.2s | - | Existing tuned LSTM settings |
| LSTM | postmortem | 0.922 | 0.721 | 0.884 | 23.1s | - | Existing tuned LSTM settings |
| Logistic Regression | alert_only | 0.963 | 0.956 | 0.968 | 0.5s | 0.018ms | TF-IDF 1-2 grams |
| Logistic Regression | early_incident | 0.969 | 0.956 | 0.972 | 0.9s | 0.024ms | TF-IDF 1-2 grams |
| Logistic Regression | postmortem | 0.944 | 0.905 | 0.947 | 2.8s | 0.073ms | TF-IDF 1-2 grams |
| Linear SVM | alert_only | 0.997 | 0.994 | 0.997 | 0.5s | 0.016ms | TF-IDF 1-2 grams |
| Linear SVM | early_incident | 0.996 | 0.993 | 0.996 | 1.0s | 0.029ms | TF-IDF 1-2 grams |
| Linear SVM | postmortem | 0.992 | 0.986 | 0.992 | 3.5s | 0.066ms | TF-IDF 1-2 grams |
| Naive Bayes | alert_only | 0.933 | 0.775 | 0.906 | 0.1s | 0.012ms | TF-IDF 1-2 grams |
| Naive Bayes | early_incident | 0.922 | 0.724 | 0.885 | 0.3s | 0.023ms | TF-IDF 1-2 grams |
| Naive Bayes | postmortem | 0.922 | 0.721 | 0.884 | 0.8s | 0.066ms | TF-IDF 1-2 grams |

The recurrent models collapse to the majority-class baseline on incident-time profiles. `RESOURCE_EXHAUSTION` accounts for 640 of 2,250 test rows, so always predicting that class yields 0.284 accuracy and very low macro F1. Adding environment, cloud provider, region, and the first two timeline events in `early_incident` did not improve GRU/LSTM performance.

The TF-IDF baselines show that the alert fields do contain strong synthetic lexical signal. Linear SVM is the best incident-time model in this experiment, reaching 0.997 accuracy and 0.994 macro F1 on `alert_only`.

The postmortem profile is dramatically easier because it includes `timeline_summary`, `root_cause_description`, and `contributing_factors`. Those fields are often discovered during or after investigation, so the high postmortem score should not be treated as live-triage performance.

## Hard Evaluation Set

The hard evaluation set is stored in `data/evaluation/hard_cases.json`. It contains 100 separately reviewed cases with ambiguity, conflicting evidence, unseen combinations, noisy wording, and missing details. These cases are not used for training or tuning.

Linear SVM hard-set results:

| Feature Profile | Cases | Accuracy | Macro F1 | Weighted F1 | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| alert_only | 100 | 0.060 | 0.051 | 0.048 | 94 |
| early_incident | 100 | 0.090 | 0.063 | 0.057 | 91 |
| postmortem | 100 | 0.080 | 0.071 | 0.069 | 92 |

Example failure themes:

- Conflicting deploy and provider evidence can be over-weighted toward deployment terms.
- Security-abuse cases with traffic symptoms can be confused with overload or resource pressure.
- Data-corruption cases described without exact training keywords are often missed.
- Dependency and cascading failures are hard to separate when the text describes both an upstream issue and downstream retries.

The gap between synthetic holdout performance and hard-set performance is the main Phase 3 finding: the TF-IDF + Linear SVM model is excellent at the synthetic dataset distribution, but it is not yet robust to realistic incident ambiguity.

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

The incident-time profiles are the more realistic live-triage benchmarks, and they currently show that TF-IDF baselines are much better suited to this synthetic alert text than GRU/LSTM models. The earlier low incident-time neural scores are a model-fit issue, not proof that the fields lack signal. A targeted class-weighted LSTM postmortem run was also tested and did not improve macro F1.

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

Train the TF-IDF baselines:

```bash
for model in logistic_regression linear_svm naive_bayes; do
  for profile in alert_only early_incident postmortem; do
    uv run python -m incident_data_classification.train_baseline --model "$model" --feature-profile "$profile" --max-rows 15000
  done
done
```
