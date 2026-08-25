# Incident RCA Classifier

Incident root-cause classifier focused on a TF-IDF + Linear SVM model for fast, leakage-aware RCA prediction from alert-time incident text. Legacy GRU/LSTM experiments are retained as neural comparison models.

## What It Does

- Downloads the synthetic RCA incident training CSV on first training run.
- Builds incident text from explicit feature profiles for alert-time, early-incident, and postmortem evaluation.
- Trains TF-IDF classical baselines and uses Linear SVM as the strongest incident-time classifier.
- Retains GRU and LSTM classifiers as legacy comparison models on the same stratified train/validation/test split.
- Evaluates accuracy, macro F1, weighted F1, and per-class precision/recall/F1.
- Exposes the Linear SVM classifier through a FastAPI service with confidence, abstention, explanations, and retrieved remediation evidence.
- Runs an interactive GRU workflow-routing demo for custom incidents.
- Runs a curated 10-sample batch demo that exercises mostly different classifications.

## Current Model Focus

The current focus is TF-IDF + Linear SVM because it performs best on the realistic incident-time profiles:

- `alert_only`: 99.7% accuracy, 99.4% macro F1, and 99.7% weighted F1 across 2,250 test rows.
- `early_incident`: 99.6% accuracy, 99.3% macro F1, and 99.6% weighted F1.
- Training finishes in about one second on CPU, with sub-millisecond inference per incident.
- The model uses sparse lexical features, which match the short categorical alert text in this synthetic dataset.

The legacy GRU/LSTM models are still useful as comparison experiments. They score highly only when the `postmortem` profile includes investigation-time fields such as root-cause description, contributing factors, and full timeline summary. On `alert_only` and `early_incident`, they collapse to the majority-class baseline. This suggests the recurrent models are not exploiting the sparse keyword and category signals that TF-IDF captures directly.

Full scorecards are documented in [README_MODEL_SCORES.md](README_MODEL_SCORES.md).

## Project Status

Portfolio-oriented ML project demonstrating lightweight incident classification, leakage-resistant evaluation, classical baseline comparison, and workflow-routing ergonomics on synthetic incident data. Production use would require real incident history, monitoring, drift detection, and human review gates.

## Dataset

Source: [RCA Synthetic Training Dataset](https://www.kaggle.com/datasets/sakthivarshans/rca-synthetic-training-dataset)

Dataset license: Apache 2.0, as listed on Kaggle at the time this project was prepared.

Training automatically downloads `incidents_training_dataset.csv` at startup if it is missing locally:

```text
data/raw/incidents_training_dataset.csv
```

Dataset and generated files are intentionally ignored by Git:

- `data/raw/*.csv`
- `models/`
- `reports/`

Hard evaluation sets are committed separately:

```text
data/evaluation/hard_cases.json
data/evaluation/real_world_hard_cases.json
```

Download only the dataset:

```bash
uv run python -m incident_data_classification.dataset
```

## Setup

Requirements:

- Python 3.11 through 3.13
- [uv](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

## Train Models

Neural training defaults to the leakage-resistant `early_incident` feature profile:

```bash
uv run python -m incident_data_classification.train --model-type gru --feature-profile early_incident
```

Supported feature profiles:

- `alert_only`: title, severity, affected services, primary affected service, and anomaly types.
- `early_incident`: `alert_only` fields plus environment, cloud provider, region, and the first two pipe-delimited timeline events.
- `postmortem`: richer comparison input that also includes timeline summary, root-cause description, and contributing factors.

Train the primary Linear SVM model:

```bash
uv run python -m incident_data_classification.train_baseline --model linear_svm --feature-profile alert_only --max-rows 15000
```

Train the DistilBERT transformer benchmark with CPU-friendly settings:

```bash
uv run python -m incident_data_classification.train_transformer --model-name distilbert-base-uncased --artifact-name distilbert --feature-profile alert_only --max-rows 5000 --epochs 2 --batch-size 16 --max-steps 300 --max-length 96
```

Train all TF-IDF baselines:

```bash
for model in logistic_regression linear_svm naive_bayes; do
  for profile in alert_only early_incident postmortem; do
    uv run python -m incident_data_classification.train_baseline --model "$model" --feature-profile "$profile" --max-rows 15000
  done
done
```

Baseline artifacts are written under `models/baselines/<model>/<feature_profile>/`, and reports are written as `reports/baseline_<model>_<feature_profile>_metrics.json`.

Train the legacy GRU profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type gru --feature-profile "$profile" --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
done
```

Train the legacy LSTM profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type lstm --feature-profile "$profile" --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
done
```

Neural artifacts are written under `models/<model>/<feature_profile>/`, and reports are written as `reports/<model>_<feature_profile>_metrics.json`.

Legacy neural tuned runs used for comparison:

```bash
uv run python -m incident_data_classification.train --model-type gru --feature-profile postmortem --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
uv run python -m incident_data_classification.train --model-type lstm --feature-profile postmortem --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
```

Training restores the best validation checkpoint before test evaluation. The default monitor is `val_macro_f1`.

For a faster smoke run:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 500 --epochs 1
```

## Evaluate

Train the profile matrix first so `models/<model>/<feature_profile>/` exists locally:

```bash
uv run python -m incident_data_classification.evaluate
```

Current saved metrics are documented in [README_MODEL_SCORES.md](README_MODEL_SCORES.md).

## Transformer Benchmark

Phase 8 adds a DistilBERT benchmark to test whether a pretrained language model improves generalization enough to justify the extra cost. It uses the same stratified train/validation/test split and the same feature-profile text construction as the other models.

Current CPU-bounded DistilBERT result:

| Model | Feature Profile | Rows | Accuracy | Macro F1 | Train Time | Inference | Model Size |
|---|---|---:|---:|---:|---:|---:|---:|
| DistilBERT | alert_only | 5,000 | 0.995 | 0.990 | 170.2s | 7.303ms | 256.1 MB |
| Linear SVM | alert_only | 15,000 | 0.997 | 0.994 | 0.5s | 0.016ms | small joblib artifact |

Hard-set comparison:

| Model | Benchmark | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| DistilBERT | Structured hard set | 0.830 | 0.778 | 0.774 |
| Linear SVM | Structured hard set | 0.910 | 0.887 | 0.881 |
| DistilBERT | Real-world prose hard set | 0.090 | 0.038 | 0.034 |
| Linear SVM | Real-world prose hard set | 0.060 | 0.051 | 0.048 |

DistilBERT can approach the synthetic Linear SVM score within a few minutes on CPU, but it does not outperform the primary model and is far more expensive to train, store, and run. The current default remains TF-IDF + Linear SVM.

## Confidence Calibration

Phase 4 adds a confidence layer for the TF-IDF baseline classifiers. For Linear SVM, raw decision margins are converted into probabilities, temperature scaling is fit on the validation split only, and calibrated confidence is evaluated on the held-out test split. A validation-selected threshold can then mark low-confidence predictions for human review.

Run calibration for the primary model:

```bash
uv run python -m incident_data_classification.evaluate_confidence --model linear_svm --feature-profile alert_only --max-rows 15000
```

Run an abstaining JSON prediction:

```bash
uv run python -m incident_data_classification.predict_baseline --model linear_svm --feature-profile alert_only --text -1
```

Current Linear SVM calibration results:

| Feature Profile | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier | Review Threshold | Accepted Accuracy | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| alert_only | 0.540 | 0.003 | 0.329 | 0.007 | 0.992 | 0.998 | 0.905 |
| early_incident | 0.525 | 0.003 | 0.313 | 0.007 | 0.991 | 0.999 | 0.905 |
| postmortem | 0.560 | 0.005 | 0.359 | 0.013 | 0.990 | 0.998 | 0.896 |

These confidence scores do not make the classifier more accurate by themselves. They make the model easier to operate: high-confidence predictions can be handled automatically, while predictions below the validation-selected threshold can be routed to a human reviewer. Reliability diagrams are generated under `reports/` during calibration evaluation.

## Similar Incident Retrieval

Phase 5 adds local similar-incident retrieval so remediation guidance is grounded in historical incidents instead of a hard-coded category-to-remedy mapping. The current implementation uses a reproducible TF-IDF 1-2 gram retrieval index with cosine similarity. Chroma, FAISS, pgvector, or Qdrant would make sense later if the project adds persistent vector-database operations or external embedding models; for the current synthetic dataset, the dependency-light TF-IDF index matches the primary classifier and is easier to reproduce.

Build retrieval indexes from the training split:

```bash
uv run python -m incident_data_classification.build_retrieval_index --feature-profile alert_only --max-rows 15000
uv run python -m incident_data_classification.build_retrieval_index --feature-profile early_incident --max-rows 15000
```

Run classification with calibrated confidence and retrieved remediation evidence:

```bash
uv run python -m incident_data_classification.predict_with_retrieval --model linear_svm --feature-profile alert_only --text -1 --top-k 3
```

Evaluate retrieval quality on a held-out review sample:

```bash
uv run python -m incident_data_classification.evaluate_retrieval --feature-profile alert_only --max-rows 15000 --sample-size 100 --top-k 3
```

Current retrieval review results:

| Feature Profile | Review Cases | Unfiltered Category Match@1 | Unfiltered Category Match@3 | Category-Filtered Mean Top-1 Similarity |
|---|---:|---:|---:|---:|
| alert_only | 100 | 0.970 | 0.990 | 0.954 |
| early_incident | 100 | 0.930 | 0.980 | 0.899 |

The production flow filters retrieved incidents by the predicted root-cause category, returns similarity scores, and names the source incident for the recommended remedy. If the classifier marks `requires_human_review: true`, the retrieved remedy should be treated as review evidence rather than an automatic remediation instruction.

## Explainability And Evidence

Phase 7 adds model evidence to the classify-plus-retrieve command. For TF-IDF baselines, the explanation reports the highest positive feature contributions for the predicted class. In practical terms, these are the words or short phrases in the incident text that pushed the Linear SVM toward its classification.

Run an explained prediction:

```bash
uv run python -m incident_data_classification.predict_with_retrieval --model linear_svm --feature-profile alert_only --text -1 --top-k 3 --top-features 8
```

The JSON output includes:

- `explanation.method`: currently `tfidf_feature_contribution`.
- `explanation.supporting_signals`: top TF-IDF terms with their feature weight and contribution.
- `evidence.model_supporting_signals`: compact list of important model features.
- `similar_incidents`: historical incidents with similarity scores and remediation evidence.
- `remedy_source`: the source incident used for the recommended remedy.

These are supporting signals, not causal proof. A high feature contribution means the term mattered to the model's decision. A similar incident means the historical text was close under the retrieval index. Neither one proves the true root cause without investigation.

## Production API

Phase 6 exposes only the TF-IDF + Linear SVM classifier. DistilBERT remains a benchmark artifact because it did not outperform Linear SVM and is much larger and slower for this dataset.

Prepare the required local artifacts:

```bash
uv run python -m incident_data_classification.train_baseline --model linear_svm --feature-profile alert_only --max-rows 15000
uv run python -m incident_data_classification.evaluate_confidence --model linear_svm --feature-profile alert_only --max-rows 15000
uv run python -m incident_data_classification.build_retrieval_index --feature-profile alert_only --max-rows 15000
```

Start the API:

```bash
uv run uvicorn incident_data_classification.api.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

```text
GET  /health
GET  /v1/model
POST /v1/incidents/classify
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/incidents/classify \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-001" \
  -d '{
    "title": "Checkout traffic spike",
    "severity": "SEV2",
    "affected_services": ["checkout-service", "api-gateway"],
    "primary_affected_service": "checkout-service",
    "anomalies": ["traffic_spike_overload", "hpa_thrashing", "high_error_rate"]
  }'
```

The response includes classification, calibrated confidence, human-review routing, alternatives, TF-IDF supporting signals, similar historical incidents, remediation evidence, model version, request ID, and inference latency.

Run with Docker:

```bash
docker compose up --build
```

The container expects local model artifacts mounted at `./models`, which remains ignored by Git.

## Operational Evaluation

Phase 1 separates model input by incident stage so evaluation is less vulnerable to post-investigation leakage.

| Model | Feature Profile | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| GRU | postmortem | 0.996 | 0.993 | 0.996 |
| GRU | early_incident | 0.284 | 0.037 | 0.126 |
| GRU | alert_only | 0.284 | 0.037 | 0.126 |
| LSTM | postmortem | 0.922 | 0.721 | 0.884 |
| LSTM | early_incident | 0.284 | 0.037 | 0.126 |
| LSTM | alert_only | 0.284 | 0.037 | 0.126 |
| Linear SVM | postmortem | 0.992 | 0.986 | 0.992 |
| Linear SVM | early_incident | 0.996 | 0.993 | 0.996 |
| Linear SVM | alert_only | 0.997 | 0.994 | 0.997 |
| Logistic Regression | postmortem | 0.944 | 0.905 | 0.947 |
| Logistic Regression | early_incident | 0.969 | 0.956 | 0.972 |
| Logistic Regression | alert_only | 0.963 | 0.956 | 0.968 |

The recurrent models collapse to the majority-class baseline on incident-time profiles, even after `early_incident` includes only the first two timeline events. The classical TF-IDF baselines perform far better on the same fields, which means the synthetic dataset exposes strong sparse lexical signals that GRU/LSTM training does not exploit well.

`postmortem` remains useful as a leakage comparison because it includes fields that often encode the answer after investigation, such as root-cause description and contributing factors. `alert_only` and `early_incident` exclude root-cause descriptions, contributing factors, remediation details, prevention recommendations, and full timeline summaries.

## Hard Evaluation

The structured hard set contains 100 curated cases with the same field shape as the training CSV, but with harder wording, distractor symptoms, and less templated descriptions. A separate real-world hard set keeps the earlier free-form incident prose benchmark. Neither hard set is used for training or tuning.

Run the hard-set benchmark against the primary Linear SVM artifact:

```bash
uv run python -m incident_data_classification.evaluate_hard_cases --model linear_svm --feature-profile alert_only
uv run python -m incident_data_classification.evaluate_hard_cases --cases data/evaluation/real_world_hard_cases.json --model linear_svm --feature-profile alert_only
```

Current Linear SVM hard-set results:

| Benchmark | Feature Profile | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| Structured hard set | alert_only | 0.910 | 0.887 | 0.881 |
| Structured hard set | early_incident | 0.910 | 0.887 | 0.881 |
| Structured hard set | postmortem | 0.830 | 0.778 | 0.774 |
| Real-world prose hard set | alert_only | 0.060 | 0.051 | 0.048 |
| Real-world prose hard set | early_incident | 0.090 | 0.063 | 0.057 |
| Real-world prose hard set | postmortem | 0.080 | 0.071 | 0.069 |

The structured hard-set result is the production-relevant framing for normalized alert payloads: when upstream systems send consistent terms such as `memory_leak`, `pod_crash_loop`, `hpa_thrashing`, or `traffic_spike_overload`, classification remains strong even with noisy surrounding context. The free-form benchmark is intentionally harsher and shows the current model is not a general RCA reasoning system. It needs stable operational terminology from monitoring, alerting, or incident enrichment systems.

## Run The Demo

Interactive GRU classifier:

```bash
uv run python -m incident_data_classification.interactive_gru --feature-profile early_incident
```

At the prompt:

- Enter an incident description to classify it.
- Enter `-1` to select one of 25 preconfigured natural-language issue descriptions.
- Enter `X` to exit.

Curated batch run:

```bash
uv run python -m incident_data_classification.interactive_gru --batch-count 10
```

The batch demo prints each message, expected class, predicted class, confidence, status, and workflow instructions.

Legacy dual-model prediction:

```bash
uv run python -m incident_data_classification.predict --text -1
```

## Architecture

```text
Dataset downloader
  -> CSV validation
  -> non-leaking text feature construction
  -> TF-IDF vectorizer + Linear SVM classifier
  -> calibrated confidence and abstention
  -> similar incident retrieval with source remediation evidence
  -> TF-IDF feature contribution evidence
  -> FastAPI service exposing the Linear SVM prediction contract
  -> baseline model artifacts and evaluation reports
  -> optional tokenizer + label encoder
  -> legacy GRU/LSTM classifier training
  -> evaluation reports
  -> GRU workflow-routing CLI
```

Primary model shape:

- `TfidfVectorizer` with word 1-2 grams
- `LinearSVC` classification head

Legacy neural model shape:

- `Embedding`
- `GRU` or `LSTM`
- `Dropout`
- linear classification head
- softmax at prediction time

The tuned GRU artifact used `embedding_dim=96`, `hidden_dim=96`, `dropout=0.25`, balanced class weights, and best-checkpoint restore.

## Validation

Local checks:

```bash
uv run pytest -q
uv run python -m incident_data_classification.evaluate
uv run python -m incident_data_classification.interactive_gru --batch-count 10
```

GitHub Actions runs `uv sync --locked` and `uv run pytest -q` on pushes, pull requests, and manual dispatch.

## Limitations

- The dataset is synthetic, so high scores can reflect synthetic generation patterns rather than real-world RCA performance.
- Workflow instructions demonstrate routing behavior. They do not execute remediation.
- The interactive natural-language samples are intentionally less model-aligned than the training data and may skew toward common classes.
- Legacy GRU/LSTM models do not perform well on incident-time profiles and are retained as comparison experiments.
- No trained model binaries, raw data, or generated reports are committed.

## License

Licensed under the [Apache License 2.0](LICENSE).
