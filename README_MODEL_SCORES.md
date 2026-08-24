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

## Confidence Calibration

Phase 4 measures confidence quality for the TF-IDF baseline path. The values below use the saved Linear SVM artifacts, fit one temperature on the validation split, and report calibration metrics on the held-out test split.

| Model | Feature Profile | Temperature | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier | Review Threshold | Coverage | Accepted Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM | alert_only | 0.266 | 0.540 | 0.003 | 0.329 | 0.007 | 0.992 | 0.905 | 0.998 |
| Linear SVM | early_incident | 0.270 | 0.525 | 0.003 | 0.313 | 0.007 | 0.991 | 0.905 | 0.999 |
| Linear SVM | postmortem | 0.259 | 0.560 | 0.005 | 0.359 | 0.013 | 0.990 | 0.896 | 0.998 |

The raw Linear SVM confidence here is produced by applying softmax to uncalibrated decision margins. That is useful for ranking classes but not suitable as an operational confidence estimate. Temperature scaling sharply reduces ECE and Brier score on this synthetic split while preserving the predicted class, because dividing all class scores by a positive temperature does not change the argmax.

The review threshold is selected from validation-set calibrated confidence at a 90% target coverage. On the test split, the accepted subset remains near that coverage and has slightly higher accuracy than the full set. This threshold is an experiment setting, not a permanent production constant.

Risk/coverage curve for Linear SVM `alert_only`:

| Target Coverage | Actual Coverage | Accepted Accuracy | Rejected Incidents |
| ---: | ---: | ---: | ---: |
| 1.00 | 1.000 | 0.997 | 0 |
| 0.90 | 0.900 | 0.998 | 225 |
| 0.80 | 0.800 | 0.999 | 450 |
| 0.70 | 0.700 | 0.999 | 675 |

## Similar Incident Retrieval

Phase 5 uses a local TF-IDF retrieval index over historical training-split incidents. The retrieved rows include source incident ID, category, cosine similarity, remediation, and prevention recommendation.

Chroma was considered, but the current project does not yet need a persistent vector database or external embedding lifecycle. TF-IDF retrieval is reproducible with the existing dependency stack, aligns with the primary Linear SVM model, and avoids introducing infrastructure before the retrieval contract is validated.

Retrieval review results on held-out test samples:

| Feature Profile | Indexed Incidents | Review Cases | Unfiltered Category Match@1 | Unfiltered Category Match@3 | Category-Filtered Mean Top-1 Similarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| alert_only | 10,500 | 100 | 0.970 | 0.990 | 0.954 |
| early_incident | 10,500 | 100 | 0.930 | 0.980 | 0.899 |

`Unfiltered Category Match@1` measures whether the nearest retrieved incident has the same root-cause category as the query without applying the production category filter. `Category-Filtered Mean Top-1 Similarity` measures the similarity of the best evidence after filtering retrieval to the predicted or expected category. This is still a synthetic-data review metric, not proof of real incident retrieval quality.

## Explainability And Evidence

Phase 7 adds TF-IDF feature contribution explanations for baseline predictions. For Linear SVM and Logistic Regression, each supporting signal is computed as:

```text
tfidf_value * predicted_class_weight
```

For Naive Bayes, the feature signal is based on class log probability relative to the average class log probability. Only positive contributions are returned as supporting signals.

The explained prediction output combines two evidence types:

- Model evidence: important TF-IDF features that pushed the classifier toward the predicted class.
- Historical evidence: retrieved similar incidents with source incident IDs, similarity scores, remediations, and prevention recommendations.

These explanations are intentionally scoped. They identify model-supporting signals and retrieved examples; they do not claim causal proof.

## Hard Evaluation Sets

Two hard benchmarks are tracked separately:

- `data/evaluation/hard_cases.json`: structured hard cases with the same field shape as the training CSV, harder wording, and distractor symptoms.
- `data/evaluation/real_world_hard_cases.json`: free-form incident prose cases with ambiguity, conflicting evidence, unseen combinations, noisy wording, and missing details.

Neither hard set is used for training or tuning.

### Structured Hard Set

| Model | Feature Profile | Cases | Accuracy | Macro F1 | Weighted F1 | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | alert_only | 100 | 0.910 | 0.887 | 0.881 | 9 |
| Logistic Regression | early_incident | 100 | 0.910 | 0.887 | 0.881 | 9 |
| Logistic Regression | postmortem | 100 | 0.900 | 0.877 | 0.871 | 10 |
| Linear SVM | alert_only | 100 | 0.910 | 0.887 | 0.881 | 9 |
| Linear SVM | early_incident | 100 | 0.910 | 0.887 | 0.881 | 9 |
| Linear SVM | postmortem | 100 | 0.830 | 0.778 | 0.774 | 17 |
| Naive Bayes | alert_only | 100 | 0.750 | 0.667 | 0.667 | 25 |
| Naive Bayes | early_incident | 100 | 0.750 | 0.667 | 0.667 | 25 |
| Naive Bayes | postmortem | 100 | 0.710 | 0.631 | 0.633 | 29 |
| GRU | alert_only | 100 | 0.080 | 0.012 | 0.012 | 92 |
| GRU | early_incident | 100 | 0.080 | 0.012 | 0.012 | 92 |
| GRU | postmortem | 100 | 0.290 | 0.222 | 0.225 | 71 |
| LSTM | alert_only | 100 | 0.080 | 0.012 | 0.012 | 92 |
| LSTM | early_incident | 100 | 0.080 | 0.012 | 0.012 | 92 |
| LSTM | postmortem | 100 | 0.110 | 0.047 | 0.050 | 89 |

The structured benchmark shows that TF-IDF models remain strong when the incoming incident payload preserves normalized alert terminology. This is the realistic integration path for a monitoring or incident-management system that emits consistent anomaly names such as `memory_leak`, `pod_crash_loop`, `hpa_thrashing`, or `traffic_spike_overload`.

### Real-World Prose Hard Set

| Model | Feature Profile | Cases | Accuracy | Macro F1 | Weighted F1 | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | alert_only | 100 | 0.060 | 0.065 | 0.065 | 94 |
| Logistic Regression | early_incident | 100 | 0.040 | 0.042 | 0.039 | 96 |
| Logistic Regression | postmortem | 100 | 0.120 | 0.051 | 0.053 | 88 |
| Linear SVM | alert_only | 100 | 0.060 | 0.051 | 0.048 | 94 |
| Linear SVM | early_incident | 100 | 0.090 | 0.063 | 0.057 | 91 |
| Linear SVM | postmortem | 100 | 0.080 | 0.071 | 0.069 | 92 |
| Naive Bayes | alert_only | 100 | 0.090 | 0.081 | 0.075 | 91 |
| Naive Bayes | early_incident | 100 | 0.090 | 0.081 | 0.075 | 91 |
| Naive Bayes | postmortem | 100 | 0.230 | 0.203 | 0.194 | 77 |
| GRU | alert_only | 100 | 0.060 | 0.009 | 0.007 | 94 |
| GRU | early_incident | 100 | 0.060 | 0.009 | 0.007 | 94 |
| GRU | postmortem | 100 | 0.060 | 0.010 | 0.007 | 94 |
| LSTM | alert_only | 100 | 0.060 | 0.009 | 0.007 | 94 |
| LSTM | early_incident | 100 | 0.060 | 0.009 | 0.007 | 94 |
| LSTM | postmortem | 100 | 0.100 | 0.062 | 0.060 | 90 |

The free-form benchmark shows the current classifiers are not robust RCA reasoning systems. Removing the normalized vocabulary causes all models to collapse, even when the prose contains enough evidence for a human reviewer.

Example failure themes:

- Conflicting deploy and provider evidence can be over-weighted toward deployment terms.
- Security-abuse cases with traffic symptoms can be confused with overload or resource pressure.
- Data-corruption cases described without exact training keywords are often missed.
- Dependency and cascading failures are hard to separate when the text describes both an upstream issue and downstream retries.

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

Evaluate the hard benchmarks:

```bash
uv run python -m incident_data_classification.evaluate_hard_cases --cases data/evaluation/hard_cases.json --model all --feature-profile alert_only
uv run python -m incident_data_classification.evaluate_hard_cases --cases data/evaluation/real_world_hard_cases.json --model all --feature-profile alert_only
```

Evaluate Linear SVM confidence calibration:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.evaluate_confidence --model linear_svm --feature-profile "$profile" --max-rows 15000
done
```

Build and review retrieval indexes:

```bash
for profile in alert_only early_incident; do
  uv run python -m incident_data_classification.build_retrieval_index --feature-profile "$profile" --max-rows 15000
  uv run python -m incident_data_classification.evaluate_retrieval --feature-profile "$profile" --max-rows 15000 --sample-size 100 --top-k 3
done
```

Run an explained classify-plus-retrieve prediction:

```bash
uv run python -m incident_data_classification.predict_with_retrieval --model linear_svm --feature-profile alert_only --text -1 --top-k 3 --top-features 8
```

Train the TF-IDF baselines:

```bash
for model in logistic_regression linear_svm naive_bayes; do
  for profile in alert_only early_incident postmortem; do
    uv run python -m incident_data_classification.train_baseline --model "$model" --feature-profile "$profile" --max-rows 15000
  done
done
```
