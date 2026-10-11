# Credit Ledger — Explainable Credit Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-EB5424.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-brightgreen.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete machine learning project for loan default prediction with model explainability (SHAP) and a clean FastAPI underwriting interface.

---

## Why I Built This

In credit scoring, accuracy alone isn't enough. If a lending model denies an applicant, regulations (like FCRA and ECOA) and basic customer trust require lenders to explain **why**. 

Traditional credit scoring often falls into one of two traps:
1. **Simple linear models** (like Logistic Regression) are easy to explain, but miss complex nonlinear relationships and underperform on edge cases.
2. **Complex ensembles** (like XGBoost or LightGBM) yield high predictive accuracy, but act like black boxes.

This project bridges that gap. I built an end-to-end pipeline that takes raw applicant data, cleans and validates it without data leakage, trains and tunes a calibrated XGBoost classifier, uses **SHAP** to explain predictions both globally across the portfolio and locally for individual applicants, and serves the model through a real-time **FastAPI** web interface.

---

## Project Highlights & Results

- **Solid Accuracy Jump**: Moving from a Logistic Regression baseline to a tuned XGBoost model pushed accuracy from **82% to 92%**.
- **Fewer False Alarms (Precision +44.6%)**: Precision jumped from **56% to 81%**, dramatically reducing the number of creditworthy applicants incorrectly rejected.
- **Strong Default Capture (Recall 81%)**: Captured 81% of actual defaulters on an unseen holdout set of 6,305 applicants.
- **Tuned Decision Threshold**: Instead of sticking with the arbitrary default cutoff of `0.5`, I tuned the decision threshold to **`0.695`** by analyzing the Precision-Recall curve to optimize the F1-score.
- **Calibrated Probabilities**: Applied Platt scaling (`CalibratedClassifierCV`) so that a predicted 30% default risk truly corresponds to ~30% empirical historical default frequency.
- **Full Explainability**: TreeSHAP feature attribution uncovers both overall portfolio risk drivers and individual waterfall charts for loan officers.

---

## Performance Comparison

Evaluated on an 80/20 stratified holdout test set (6,305 applicants):

| Model | Accuracy | Precision | Recall | F1 Score | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Logistic Regression) | 82.0% | 56.0% | 79.0% | 0.65 | 0.871 |
| **XGBoost (Tuned & Calibrated)** | **92.0%** | **81.0%** | **81.0%** | **0.81** | **0.939** |

### Confusion Matrix on Test Set:
- **True Negatives**: 4,685 (correctly approved safe borrowers)
- **True Positives**: 1,106 (correctly flagged defaults)
- **False Positives**: 258 (safe applicants wrongly flagged risky — down from 840+ in baseline)
- **False Negatives**: 256 (missed defaults)

---

## How It Works

### 1. Data Cleaning & Sanity Checks
The dataset contains 32,581 loan records across 12 features. Real-world lending data always has anomalies, so I filtered:
- Impossible ages (e.g., records showing age 144)
- Employment lengths greater than applicant age or > 60 years
- Non-positive incomes and loan amounts
- Duplicate applicant entries (165 records)

### 2. Handling Class Imbalance
About **78%** of applicants in the dataset repaid their loans and **22%** defaulted (~3.58 : 1 ratio). Rather than artificially synthesizing samples with SMOTE (which can distort real financial distributions), I calculated a dynamic `scale_pos_weight` and passed it directly to XGBoost to penalize missed defaults during training.

### 3. Pipeline Architecture
To prevent any data leakage during cross-validation, all transformations live inside Scikit-learn `Pipeline` and `ColumnTransformer` objects:
- **Numerical columns**: Median imputation (and `StandardScaler` for Logistic Regression; scale-invariant for XGBoost).
- **Categorical columns**: Most-frequent imputation + One-Hot Encoding (`handle_unknown='ignore'`).

### 4. Hyperparameter Tuning
Ran a `RandomizedSearchCV` with 5-fold Stratified K-Fold across tree depth, learning rate, subsample ratio, colsample ratio, and minimum child weight. The best estimator settled at `max_depth=5`, `learning_rate=0.13`, and `n_estimators=366`.

### 5. Probability Calibration
Raw tree model probabilities can be overconfident or poorly calibrated. I plotted calibration curves and fitted a Sigmoid calibrator (`CalibratedClassifierCV`) so the output scores represent dependable risk probabilities.

### 6. SHAP Interpretability
- **Global Insights**: The top drivers of default risk across the dataset are `loan_percent_income` (debt-to-income burden), prior defaults on credit bureau file (`cb_person_default_on_file`), assigned `loan_grade`, and interest rates.
- **Local Waterfall Plots**: For any single loan application, we can generate a waterfall plot showing the exact dollar and percentage contributions pushing their score above or below the baseline threshold.

---

## The Web App & API

I built a lightweight underwriting console using **FastAPI** and vanilla JS/CSS:
- **Underwriting UI**: Allows loan officers to input applicant details and immediately view the calculated default probability, a visual gauge, and an underwriting verdict (`LOW RISK` / `HIGH RISK`).
- **Input Validation**: Backed by Pydantic schemas with type enforcement and realistic bounds.
- **Auto Ratio Calculation**: Dynamically computes loan-to-income ratio as the officer types.
- **Interactive Docs**: Swagger UI available at `/docs`.

---

## Project Structure

```
.
├── static/
│   ├── index.html              # Underwriting desk interface
│   ├── script.js               # Form handling, animated gauge, API calls
│   └── style.css               # Clean styling
├── Credit_Risk.ipynb           # Complete research notebook (EDA -> Tuning -> SHAP)
├── credit_risk_dataset.csv     # Historical loan applicant dataset
├── credit_risk_model.pkl       # Trained & calibrated model pipeline
├── best_threshold.pkl          # Pickled decision cutoff (0.695)
├── main.py                     # FastAPI backend
├── requirements.txt            # Dependencies
├── runtime.txt                 # Target Python version (3.11.9)
├── render.yaml                 # One-click Render deployment config
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/ishanmane0/Credit-Risk-Assessment-using-SHAP.git
cd Credit-Risk-Assessment-using-SHAP
```

### 2. Set up virtual environment
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
uvicorn main:app --reload
```
Open **`http://localhost:8000`** in your browser to see the underwriting desk, or **`http://localhost:8000/docs`** for the API documentation.

### 5. Open the notebook
To explore the analysis and SHAP plots:
```bash
jupyter notebook Credit_Risk.ipynb
```

---

## API Endpoints

- **`GET /health`** — Basic health check and model loading status.
- **`POST /predict`** — Predict default risk for an applicant.

**Sample Request (`POST /predict`):**
```json
{
  "person_age": 30,
  "person_income": 600000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 100000.0,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Sample Response:**
```json
{
  "default_probability": 0.1245,
  "default_prediction": 0,
  "threshold": 0.6952,
  "Result": "Low Risk"
}
```

---

## License

MIT
