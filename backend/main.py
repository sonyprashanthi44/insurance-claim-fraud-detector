"""FastAPI service for insurance claim fraud scoring."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from llm_explain import explain_claim
except ImportError:  # uvicorn backend.main:app
    from backend.llm_explain import explain_claim

ART_DIR = Path(__file__).resolve().parent / "artifacts"

app = FastAPI(
    title="Insurance Claim Fraud Detector",
    version="1.0.0",
    description="Screening aid for vehicle insurance claim fraud risk.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup
BUNDLE: dict[str, Any] = {}
METRICS: dict[str, Any] = {}
PR_CURVE: dict[str, Any] = {}
THRESHOLDS: dict[str, Any] = {}
SAMPLES: list[dict[str, Any]] = []


class ClaimFeatures(BaseModel):
    """Raw claim fields accepted by /predict and /explain."""

    Month: str | None = None
    WeekOfMonth: int | str | None = None
    DayOfWeek: str | None = None
    Make: str | None = None
    AccidentArea: str | None = None
    DayOfWeekClaimed: str | None = None
    MonthClaimed: str | None = None
    WeekOfMonthClaimed: int | str | None = None
    Sex: str | None = None
    MaritalStatus: str | None = None
    Age: float | int | None = None
    Fault: str | None = None
    PolicyType: str | None = None
    VehicleCategory: str | None = None
    VehiclePrice: str | None = None
    Deductible: int | str | None = None
    DriverRating: int | str | None = None
    Days_Policy_Accident: str | None = None
    Days_Policy_Claim: str | None = None
    PastNumberOfClaims: str | None = None
    AgeOfVehicle: str | None = None
    PoliceReportFiled: str | None = None
    WitnessPresent: str | None = None
    AgentType: str | None = None
    NumberOfSuppliments: str | None = None
    AddressChange_Claim: str | None = None
    NumberOfCars: str | None = None
    Year: int | str | None = None
    BasePolicy: str | None = None

    model_config = {"extra": "allow"}


class Contribution(BaseModel):
    feature: str
    value: Any
    shap: float


class PredictResponse(BaseModel):
    probability: float
    band: str
    top_contributions: list[Contribution]


class ExplainRequest(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    band: str
    top_contributions: list[Contribution]
    claim: dict[str, Any] | None = None


class ExplainResponse(BaseModel):
    explanation: str
    source: str


def band_for_proba(p: float, thresholds: dict[str, Any]) -> str:
    if p >= float(thresholds["high"]):
        return "High"
    if p >= float(thresholds["medium"]):
        return "Medium"
    return "Low"


def _claim_to_frame(claim: dict[str, Any]) -> pd.DataFrame:
    cols: list[str] = list(BUNDLE.get("raw_feature_columns") or [])
    row = {}
    for c in cols:
        row[c] = claim.get(c, None)
    # Also accept extras that might match
    for k, v in claim.items():
        if k not in row and k in cols:
            row[k] = v
    df = pd.DataFrame([row], columns=cols)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df.loc[df["Age"] <= 0, "Age"] = np.nan
    for c in df.columns:
        if c == "Age":
            continue
        df[c] = df[c].apply(lambda x: None if x is None else str(x))
    return df


def _pretty_feature(name: str) -> str:
    # ColumnTransformer prefixes: num__, ord__, cat__
    for prefix in ("num__", "ord__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name


def _raw_value_for_feature(feat_name: str, claim: dict[str, Any]) -> Any:
    pretty = _pretty_feature(feat_name)
    base = pretty.split("=")[0] if "=" in pretty else pretty
    if base in claim:
        return claim[base]
    # one-hot style cat__Make_Honda
    for k, v in claim.items():
        if pretty.startswith(k):
            return v
    return claim.get(base, None)


def score_claim(claim: dict[str, Any]) -> PredictResponse:
    pipe = BUNDLE["pipeline"]
    explainer = BUNDLE["explainer"]
    feat_names: list[str] = BUNDLE["feature_names"]
    thresholds = BUNDLE["thresholds"]

    X = _claim_to_frame(claim)
    proba = float(pipe.predict_proba(X)[0, 1])
    band = band_for_proba(proba, thresholds)

    pre = pipe.named_steps["preprocess"]
    Xt = pre.transform(X)
    sv = explainer.shap_values(Xt)
    # LightGBM binary: list [class0, class1] or 2D array
    if isinstance(sv, list):
        shap_row = np.array(sv[1][0], dtype=float)
    else:
        arr = np.array(sv, dtype=float)
        shap_row = arr[0] if arr.ndim == 2 else arr

    order = np.argsort(-np.abs(shap_row))[:6]
    contribs: list[Contribution] = []
    for i in order:
        fname = feat_names[int(i)] if int(i) < len(feat_names) else f"f{i}"
        contribs.append(
            Contribution(
                feature=_pretty_feature(fname),
                value=_raw_value_for_feature(fname, claim),
                shap=float(shap_row[int(i)]),
            )
        )
    return PredictResponse(probability=proba, band=band, top_contributions=contribs)


@app.on_event("startup")
def load_artifacts() -> None:
    model_path = ART_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Missing {model_path}. Run: python backend/train.py"
        )
    global BUNDLE, METRICS, PR_CURVE, THRESHOLDS, SAMPLES
    BUNDLE = joblib.load(model_path)
    with open(ART_DIR / "metrics.json", encoding="utf-8") as f:
        METRICS = json.load(f)
    with open(ART_DIR / "pr_curve.json", encoding="utf-8") as f:
        PR_CURVE = json.load(f)
    with open(ART_DIR / "thresholds.json", encoding="utf-8") as f:
        THRESHOLDS = json.load(f)
    with open(ART_DIR / "sample_claims.json", encoding="utf-8") as f:
        SAMPLES = json.load(f)
    # Prefer thresholds embedded with the model bundle
    if "thresholds" in BUNDLE:
        THRESHOLDS = BUNDLE["thresholds"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return {
        "metrics": METRICS,
        "pr_curve": PR_CURVE,
        "thresholds": THRESHOLDS,
    }


@app.get("/samples")
def get_samples() -> dict[str, Any]:
    return {"samples": SAMPLES}


@app.post("/predict", response_model=PredictResponse)
def predict(claim: ClaimFeatures) -> PredictResponse:
    try:
        payload = claim.model_dump()
        return score_claim(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Predict failed: {exc}") from exc


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest) -> ExplainResponse:
    result = explain_claim(
        probability=body.probability,
        band=body.band,
        contributions=[c.model_dump() for c in body.top_contributions],
        claim=body.claim,
    )
    return ExplainResponse(**result)
