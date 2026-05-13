# Diabetes Risk Predictor

A TripleTen final project that combines a trained ML classification model with an LLM-powered natural language interface. Users describe a patient in plain English; the system extracts the relevant clinical features, runs them through the trained model, and returns a plain-English explanation of the prediction.

## Problem

Predict the probability of diabetes onset using the Pima Indians Diabetes dataset (768 patients, 8 clinical features). Binary classification: `0` = no diabetes, `1` = diabetes.

## Who It Is For

Healthcare students or educators who want to explore how ML predictions can be made accessible through natural language — without needing to fill in a form or understand the underlying features.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| ML models | scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| LLM interface | Nebius AI Studio (OpenAI-compatible) |
| App UI | CLI (interactive loop) |
| Tests | pytest |

## Project Structure

```
diabetes_prediction/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   └── config.yaml          # Training hyperparameters and settings
├── src/
│   ├── preprocess.py        # Data cleaning and feature pipeline
│   ├── train.py             # MLflow training script
│   ├── evaluate.py          # Model evaluation utilities
│   └── app.py               # CLI + LLM interface
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_interface.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
└── data/
    └── .gitkeep              # Data not committed — add diabetes.csv here
```

---

## Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Edit .env and add your NEBIUS_API_KEY

# 3. Add the dataset
# Download diabetes.csv from Kaggle (Pima Indians Diabetes Dataset)
# and place it in the data/ folder

# 4. Train all models (logs 6 runs to MLflow)
python src/train.py

# 5. Run the app
python src/app.py
```

## Usage

Once the app is running, describe a patient in plain English:

```
You: 45-year-old woman, BMI 32, fasting glucose 148, diastolic BP 80, 2 pregnancies
Risk Level : High Risk (71.3% probability of diabetes)
Assistant  : Based on the provided information, the model predicts a high risk...
```

- If features are missing, the model substitutes median training values and notifies you
- If fewer than 3 features are provided, the app asks for more information
- If the input is unrelated to patient data, the app redirects you

## Docker (optional)

```bash
docker build -t diabetes-predictor .
docker run -it -e NEBIUS_API_KEY=your_key_here diabetes-predictor
```

## MLflow

Browse all experiment runs:
```bash
mlflow ui --backend-store-uri mlruns --port 5011
```
Then open `http://localhost:5011`.

## Tests

```bash
pytest tests/
```

---

## Architecture Overview

```
User input (plain English)
        │
        ▼
  LLM (Nebius) — parse_features()
  Extracts 8 clinical values from text → JSON
        │
        ▼
  sklearn Pipeline — run_inference()
  Median imputer → StandardScaler → trained classifier
        │
        ▼
  Prediction (0/1) + probability
        │
        ▼
  LLM (Nebius) — explain_prediction()
  Generates plain-English explanation of the result
        │
        ▼
  CLI output
```

The LLM is called twice per query: once to parse the input into structured features, and once to explain the model's output. The ML model itself is a scikit-learn `Pipeline` trained offline and loaded from MLflow at startup.

---

## Results Summary

Six model configurations were trained and tracked with MLflow. Best model selected by ROC-AUC.

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **Random Forest** ✓ | **0.8512** | **0.5610** | **0.7616** |
| Logistic Regression | 0.8500 | 0.5432 | 0.7550 |
| Logistic Regression v2 | — | — | — |
| Random Forest v2 | — | — | — |
| XGBoost | 0.8246 | 0.5843 | 0.7550 |
| XGBoost v2 | — | — | — |

**Best model: Random Forest** (200 trees, max depth 8)
- ROC-AUC: 0.8512
- F1: 0.5610
- Precision: 0.7931 · Recall: 0.4340 · Accuracy: 0.7616

**Key finding:** All three algorithms scored within 0.03 ROC-AUC of each other. The dataset is small (752 rows after cleaning) so differences are not statistically significant. High precision but low recall suggests the model is conservative — it misses some true positives but rarely raises a false alarm.

---

## Preprocessing Summary

| Step | Decision |
|------|----------|
| Sentinel zeros | Replaced with `NaN` (biologically impossible values) |
| Dropped rows | Only rows missing Glucose or BMI |
| Remaining NaN | Median imputation inside the pipeline |
| Scaling | `StandardScaler` on all features |
| Train/test split | 80/20 stratified |

---

## Reflection

**What I learned:**
- How to wrap an ML model inside a full Pipeline so that preprocessing is never applied to test data during training (preventing data leakage)
- How to use MLflow to track multiple experiment runs and programmatically select the best one with `search_runs()`
- How to use an LLM as a parsing layer — letting users interact in natural language while the model still receives structured input

**What was challenging:**
- Handling missing values correctly across the full pipeline (sentinel zeros → NaN → median imputation inside sklearn)
- Getting the LLM to reliably return valid JSON and handling cases where it adds markdown fences
- Deciding when "not enough information" means asking a clarifying question vs. proceeding with median substitution

**What I would improve with more time:**
- Add a confidence threshold: if probability is between 40–60%, flag it as uncertain rather than committing to high/low risk
- Collect more data — 752 rows is very small for a medical dataset; recall of 0.43 means the model misses more than half of true diabetes cases
- Try feature engineering (e.g. glucose × BMI interaction) which may improve recall without sacrificing precision
