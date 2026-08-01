# CodeAlpha_CreditScoringModel

**Task 1 — Machine Learning Internship, CodeAlpha**

Predicts an individual's creditworthiness (good credit vs. bad-credit / default risk) from financial history, using classic classification algorithms and served through a REST API with a live demo UI.

## Objective

> Predict an individual's creditworthiness using past financial data.

## Overview

This project goes beyond a single notebook — it's a complete, end-to-end pipeline:

```
data → preprocessing → model training/comparison → best-model selection → REST API → demo UI
```

| Layer | What it does |
|---|---|
| `src/preprocess.py` | Loads the raw dataset, maps coded categories to human-readable labels, engineers financial-history features |
| `src/train.py` | Trains Logistic Regression, Decision Tree, and Random Forest; evaluates each on Precision, Recall, F1-Score, and ROC-AUC; saves the best model |
| `api/main.py` | FastAPI service exposing `/predict` — a live scoring endpoint |
| `frontend/index.html` | Single-page demo UI that calls the API and renders a risk verdict |
| `outputs/` | Model comparison table, ROC curve plot, feature importance plot |

## Dataset

[Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — UCI Machine Learning Repository. 1,000 loan applicants, 20 attributes (income, savings, employment history, existing debts, credit history, purpose, etc.), labeled as good or bad credit risk.

## Approach

- **Feature engineering**: debt-to-duration ratio, foreign-worker flag, "no checking account" flag, "critical credit history" flag — derived from raw financial-history fields.
- **Models compared**: Logistic Regression, Decision Tree, Random Forest (all with `class_weight="balanced"` since defaults are the minority class, ~30%).
- **Evaluation metrics**: Precision, Recall, F1-Score, ROC-AUC — reported per model, with confusion matrices and an ROC curve overlay.
- **Model selection**: best model by ROC-AUC is auto-saved to `models/best_model.pkl` for serving.

### Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.65 | 0.58 | 0.61 | **0.81** |
| Logistic Regression | 0.56 | 0.82 | 0.66 | 0.81 |
| Decision Tree | 0.42 | 0.63 | 0.50 | 0.62 |

Random Forest wins on ROC-AUC and is auto-selected as the serving model. Logistic Regression is worth noting too — it has the highest recall on bad-credit cases (0.82), meaning it catches more actual defaulters, which matters more than precision in many real lending risk policies. See `outputs/model_comparison.csv` and `outputs/roc_curves.png` for full results.

## How to Run

### 1. Setup
```bash
git clone https://github.com/<your-username>/CodeAlpha_CreditScoringModel.git
cd CodeAlpha_CreditScoringModel
pip install -r requirements.txt
```

### 2. Train the models
```bash
cd src
python train.py
```
This regenerates `models/best_model.pkl` and everything in `outputs/`.

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or `POST /predict` directly.

### 4. Open the demo UI
Open `frontend/index.html` in a browser (with the API running) to fill in an applicant's details and get a live risk score.

## Project Structure
```
CodeAlpha_CreditScoringModel/
├── api/
│   └── main.py              # FastAPI serving layer
├── data/
│   └── german_credit.csv    # raw Statlog dataset
├── frontend/
│   └── index.html           # demo UI
├── models/
│   ├── best_model.pkl
│   └── best_model_info.json
├── outputs/
│   ├── model_comparison.csv
│   ├── roc_curves.png
│   └── feature_importance.png
├── src/
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Tech Stack
Python · pandas · scikit-learn · matplotlib · FastAPI · Uvicorn · HTML/CSS/JS

## Author
Shubhga Mudgal — B.Tech CSE (Final Year)
[LinkedIn](https://www.linkedin.com/in/shubhgamudgal-020596298) · [GitHub](https://github.com/shubhga)

## Acknowledgment
Built as part of the **CodeAlpha Machine Learning Internship** — Task 1: Credit Scoring Model.
Dataset: Prof. Hans Hofmann, Statlog (German Credit Data), UCI Machine Learning Repository.
