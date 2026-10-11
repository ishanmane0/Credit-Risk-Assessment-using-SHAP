from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# In-memory model artifact container
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager that loads ML models and artifacts at startup
    and gracefully releases resources at shutdown.
    """
    ml_model["model"] = joblib.load("credit_risk_model.pkl")
    ml_model["threshold"] = joblib.load("best_threshold.pkl")
    yield
    ml_model.clear()


app = FastAPI(
    title="Credit Risk Assessment & Underwriting API",
    description=(
        "Production-ready scoring service using calibrated XGBoost and SHAP "
        "interpretability for loan default probability estimation and decisioning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients and cross-origin underwriting desks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):
    """Schema representing loan applicant data and credit bureau indicators."""

    person_age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age in years",
        examples=[30],
    )
    person_income: float = Field(
        ...,
        gt=0,
        description="Annual income in currency units",
        examples=[600000.0],
    )
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"] = Field(
        ...,
        description="Home ownership status",
        examples=["RENT"],
    )
    person_emp_length: float = Field(
        ...,
        ge=0,
        le=60,
        description="Employment duration in years",
        examples=[5.0],
    )
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ] = Field(
        ...,
        description="Stated purpose of the loan",
        examples=["PERSONAL"],
    )
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ...,
        description="Assigned risk credit grade",
        examples=["B"],
    )
    loan_amnt: float = Field(
        ...,
        gt=0,
        description="Requested loan principal amount",
        examples=[100000.0],
    )
    loan_int_rate: float = Field(
        ...,
        gt=0,
        le=40,
        description="Annual loan interest rate (percentage)",
        examples=[11.5],
    )
    loan_percent_income: float = Field(
        ...,
        ge=0,
        le=1,
        description="Loan amount relative to annual income (ratio)",
        examples=[0.17],
    )
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        ...,
        description="Historical default record on credit bureau file ('Y' or 'N')",
        examples=["N"],
    )
    cb_person_cred_hist_length: int = Field(
        ...,
        ge=0,
        le=60,
        description="Length of credit bureau history in years",
        examples=[6],
    )


class PredictionResponse(BaseModel):
    """Response schema containing risk probability and underwriting verdict."""

    default_probability: float = Field(
        ..., description="Calibrated probability of loan default [0.0, 1.0]"
    )
    default_prediction: int = Field(
        ..., description="Binary classification (1 = High Risk / Default, 0 = Low Risk)"
    )
    threshold: float = Field(
        ..., description="Decision boundary threshold applied to probability"
    )
    Result: str = Field(
        ..., description="Underwriting verdict ('High Risk' or 'Low Risk')"
    )


@app.get("/health", tags=["System"])
def health_check():
    """Service health inspection and artifact availability check."""
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_model,
        "decision_threshold": ml_model.get("threshold", None),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Underwriting"],
    summary="Assess Loan Default Risk",
)
def predict(data: LoanApplication):
    """
    Evaluates loan applicant risk profile using calibrated machine learning pipeline
    and returns estimated default probability alongside an underwriting decision.
    """
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    input_df = pd.DataFrame([payload])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    threshold = float(ml_model["threshold"])
    prediction = int(probability >= threshold)

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Mount web interface static assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")
