# Model Scores

These results come from the first small training run on `incidents_training_dataset.csv`.

Run configuration:

- Rows used: 3,000 stratified rows
- Test rows: 450
- Epochs: 3
- Batch size: 64
- Vocabulary size limit: 8,000
- Sequence length: 96 tokens
- Embedding dimension: 64
- Hidden dimension: 64
- Device: CPU

## Overall Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Train Time |
| --- | ---: | ---: | ---: | ---: |
| GRU | 0.869 | 0.647 | 0.832 | 2.2s |
| LSTM | 0.796 | 0.511 | 0.739 | 2.7s |

Macro F1 is the better comparison number here because the classes are imbalanced. On this small run, the GRU is ahead overall and across most classes.

## GRU Per-Class Scores

| Classification | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| CASCADING_FAILURE | 1.00 | 0.95 | 0.97 | 19 |
| CONFIGURATION_ERROR | 1.00 | 0.70 | 0.82 | 20 |
| DATA_CORRUPTION | 0.00 | 0.00 | 0.00 | 11 |
| DEPENDENCY_FAILURE | 0.90 | 1.00 | 0.95 | 84 |
| DEPLOYMENT_REGRESSION | 0.85 | 0.98 | 0.91 | 41 |
| INFRASTRUCTURE_FAILURE | 0.78 | 0.93 | 0.84 | 41 |
| RESOURCE_EXHAUSTION | 0.92 | 1.00 | 0.96 | 128 |
| RESOURCE_LEAK | 0.75 | 1.00 | 0.86 | 43 |
| SCHEDULED_JOB_FAILURE | 0.00 | 0.00 | 0.00 | 12 |
| SECURITY_INCIDENT | 0.70 | 0.70 | 0.70 | 20 |
| THIRD_PARTY_FAILURE | 0.00 | 0.00 | 0.00 | 12 |
| TRAFFIC_OVERLOAD | 0.92 | 0.63 | 0.75 | 19 |

## LSTM Per-Class Scores

| Classification | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| CASCADING_FAILURE | 1.00 | 0.37 | 0.54 | 19 |
| CONFIGURATION_ERROR | 1.00 | 0.10 | 0.18 | 20 |
| DATA_CORRUPTION | 0.00 | 0.00 | 0.00 | 11 |
| DEPENDENCY_FAILURE | 0.83 | 1.00 | 0.91 | 84 |
| DEPLOYMENT_REGRESSION | 0.84 | 0.90 | 0.87 | 41 |
| INFRASTRUCTURE_FAILURE | 0.62 | 1.00 | 0.77 | 41 |
| RESOURCE_EXHAUSTION | 0.86 | 1.00 | 0.92 | 128 |
| RESOURCE_LEAK | 0.71 | 0.98 | 0.82 | 43 |
| SCHEDULED_JOB_FAILURE | 0.00 | 0.00 | 0.00 | 12 |
| SECURITY_INCIDENT | 0.67 | 0.40 | 0.50 | 20 |
| THIRD_PARTY_FAILURE | 0.00 | 0.00 | 0.00 | 12 |
| TRAFFIC_OVERLOAD | 0.90 | 0.47 | 0.62 | 19 |

## Current Read

Both models are already strong on common classes like `RESOURCE_EXHAUSTION`, `DEPENDENCY_FAILURE`, `DEPLOYMENT_REGRESSION`, and `RESOURCE_LEAK`.

Both models are failing on smaller classes in this run:

- `DATA_CORRUPTION`
- `SCHEDULED_JOB_FAILURE`
- `THIRD_PARTY_FAILURE`

That is expected with only 3,000 sampled rows, 3 epochs, and class imbalance. The next useful experiment is to train on more rows or add class weighting so the smaller categories are not ignored.

## Commands

Recreate this kind of small run:

```bash
uv run python -m incident_data_classification.train --model-type gru --max-rows 3000 --epochs 3
uv run python -m incident_data_classification.train --model-type lstm --max-rows 3000 --epochs 3
uv run python -m incident_data_classification.evaluate
```

