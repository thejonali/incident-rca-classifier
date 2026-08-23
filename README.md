# Incident RCA Classifier

PyTorch GRU/LSTM classifiers for mapping incident descriptions to root-cause categories, with a CLI demo that routes the predicted class to workflow instructions.

## What It Does

- Downloads the synthetic RCA incident training CSV on first training run.
- Builds incident text from explicit feature profiles for alert-time, early-incident, and postmortem evaluation.
- Trains GRU and LSTM classifiers on the same stratified train/validation/test split.
- Evaluates accuracy, macro F1, weighted F1, and per-class precision/recall/F1.
- Runs an interactive GRU workflow-routing demo for custom incidents.
- Runs a curated 10-sample batch demo that exercises mostly different classifications.

## GRU Highlights

The tuned GRU is the strongest model in the project on the postmortem synthetic holdout split:

- 99.6% accuracy, 99.3% macro F1, and 99.6% weighted F1 across 2,250 test rows.
- Above 95% F1 for every root-cause category.
- Balanced class weights improved minority-class performance across categories such as `DATA_CORRUPTION`, `SCHEDULED_JOB_FAILURE`, and `THIRD_PARTY_FAILURE`.
- Trained in about 125 seconds on CPU with `embedding_dim=96`, `hidden_dim=96`, and `dropout=0.25`.

Full GRU and LSTM scorecards are documented in [README_MODEL_SCORES.md](README_MODEL_SCORES.md).

## Project Status

Portfolio-oriented ML project demonstrating lightweight incident classification, model comparison, and workflow-routing ergonomics on synthetic incident data. Production use would require real incident history, monitoring, drift detection, and human review gates.

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

Training defaults to the leakage-resistant `early_incident` feature profile:

```bash
uv run python -m incident_data_classification.train --model-type gru --feature-profile early_incident
```

Supported feature profiles:

- `alert_only`: title, severity, affected services, primary affected service, and anomaly types.
- `early_incident`: `alert_only` fields plus environment, cloud provider, and region.
- `postmortem`: richer comparison input that also includes timeline summary, root-cause description, and contributing factors.

Train the reported Phase 1 GRU profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type gru --feature-profile "$profile" --max-rows 15000 --epochs 20 --batch-size 64 --vocab-size 12000 --max-length 128 --embedding-dim 96 --hidden-dim 96 --class-weights balanced --patience 5
done
```

Train the reported Phase 1 LSTM profile matrix:

```bash
for profile in alert_only early_incident postmortem; do
  uv run python -m incident_data_classification.train --model-type lstm --feature-profile "$profile" --max-rows 15000 --epochs 12 --batch-size 64 --vocab-size 10000 --max-length 96 --embedding-dim 64 --hidden-dim 64 --learning-rate 0.003 --patience 4
done
```

Artifacts are written under `models/<model>/<feature_profile>/`, and reports are written as `reports/<model>_<feature_profile>_metrics.json`.

Tuned runs used for the current reported scores:

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

The incident-time profiles collapse to the majority-class baseline on this synthetic dataset. `postmortem` is much higher because the model can see fields that often encode the answer after investigation, such as root-cause description and contributing factors. `alert_only` and `early_incident` are more realistic for live triage because they exclude root-cause descriptions, contributing factors, remediation details, prevention recommendations, and full timeline summaries. These lower scores are more useful for estimating real incident-time decision support.

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
  -> tokenizer + label encoder
  -> GRU/LSTM classifier training
  -> saved model artifacts
  -> evaluation reports
  -> GRU workflow-routing CLI
```

Model shape:

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
- The LSTM clears 90% accuracy but has much weaker macro F1 than the GRU because it misses several smaller classes.
- No trained model binaries, raw data, or generated reports are committed.

## License

Licensed under the [Apache License 2.0](LICENSE).
