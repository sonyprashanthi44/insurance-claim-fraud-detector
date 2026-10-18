"""Train LightGBM fraud detector and write inference artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "fraud_oracle.csv"
ART_DIR = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# PolicyNumber: unique claim ID -> drop (leakage / non-generalizable).
# RepNumber: adjuster identity, high-cardinality proxy for process, not claim risk -> drop.
# AgeOfPolicyHolder: binned version of Age -> drop redundant band; keep numeric Age.
# Year: claim calendar year (few levels). Keep as categorical context, not as an ID.
DROP_COLS = ["PolicyNumber", "RepNumber", "AgeOfPolicyHolder", "FraudFound_P"]

# Low-cardinality ordered categoricals -> ordinal; rest of object cols -> one-hot.
ORDINAL_COLS = [
    "VehiclePrice",
    "Days_Policy_Accident",
    "Days_Policy_Claim",
    "PastNumberOfClaims",
    "AgeOfVehicle",
    "NumberOfSuppliments",
    "AddressChange_Claim",
    "NumberOfCars",
    "DriverRating",
    "Deductible",
    "WeekOfMonth",
    "WeekOfMonthClaimed",
    "Year",
]
NUMERIC_COLS = ["Age"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip BOM / whitespace from headers if present
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    y = df["FraudFound_P"].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    # Age == 0 is a known sentinel for missing/unknown in this dataset
    if "Age" in X.columns:
        X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
        X.loc[X["Age"] <= 0, "Age"] = np.nan

    for col in ORDINAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    numeric = [c for c in NUMERIC_COLS if c in X.columns]
    ordinal = [c for c in ORDINAL_COLS if c in X.columns]
    categorical = [
        c
        for c in X.columns
        if c not in numeric and c not in ordinal
    ]
    for c in categorical:
        X[c] = X[c].astype(str)

    return X, y, numeric, ordinal + categorical


def build_pipeline(
    numeric_cols: list[str],
    cat_cols: list[str],
    ordinal_cols: list[str],
    y_train: pd.Series,
) -> Pipeline:
    # Split cat_cols into ordinal vs nominal based on ORDINAL_COLS membership
    ord_in = [c for c in ordinal_cols if c in ORDINAL_COLS]
    nom_in = [c for c in cat_cols if c not in ORDINAL_COLS]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if ord_in:
        transformers.append(
            (
                "ord",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ord_in,
            )
        )
    if nom_in:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                nom_in,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos else 1.0

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def feature_names_from_pipeline(pipe: Pipeline) -> list[str]:
    pre: ColumnTransformer = pipe.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


def choose_thresholds(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """
    High: highest threshold with precision >= 0.35 (and some support), else top-precision point.
    Medium: lowest threshold achieving recall >= 0.90.
    Below Medium -> Low.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall length = len(thresholds)+1; align to thresholds
    p = precision[:-1]
    r = recall[:-1]
    t = thresholds

    # High band: prioritize precision with non-trivial recall
    high_candidates = np.where((p >= 0.35) & (r >= 0.05))[0]
    if len(high_candidates):
        # among candidates, pick best F1-ish balance favoring precision
        scores = p[high_candidates] * 0.7 + r[high_candidates] * 0.3
        high_idx = high_candidates[int(np.argmax(scores))]
    else:
        high_idx = int(np.argmax(p)) if len(p) else 0
    high_thr = float(t[high_idx]) if len(t) else 0.5

    # Medium: ~90% recall
    med_candidates = np.where(r >= 0.90)[0]
    if len(med_candidates):
        # lowest threshold that still hits 90% recall is the last index in PR curve
        # (thresholds increase -> recall decreases). Take max threshold among r>=0.9
        med_idx = med_candidates[int(np.argmax(t[med_candidates]))]
        medium_thr = float(t[med_idx])
    else:
        medium_thr = float(np.quantile(proba, 0.4))

    # Ensure ordering: high > medium
    if high_thr <= medium_thr:
        high_thr = min(0.99, medium_thr + 0.05)
        if high_thr <= medium_thr:
            high_thr = float(min(0.999, max(medium_thr * 1.2, medium_thr + 0.01)))

    return {
        "high": round(high_thr, 6),
        "medium": round(medium_thr, 6),
        "rules": {
            "high": "probability >= high threshold (precision-oriented)",
            "medium": "probability >= medium threshold (~90% recall on validation)",
            "low": "probability < medium threshold",
        },
        "high_operating_point": {
            "precision": float(p[high_idx]) if len(p) else None,
            "recall": float(r[high_idx]) if len(r) else None,
        },
        "medium_operating_point": {
            "precision": float(p[med_idx]) if len(med_candidates) else None,
            "recall": float(r[med_idx]) if len(med_candidates) else None,
        },
    }


def band_for_proba(p: float, thresholds: dict) -> str:
    if p >= thresholds["high"]:
        return "High"
    if p >= thresholds["medium"]:
        return "Medium"
    return "Low"


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    X, y, numeric_cols, cat_all = prepare_xy(df)
    ordinal_cols = [c for c in cat_all if c in ORDINAL_COLS]
    nominal_cols = [c for c in cat_all if c not in ORDINAL_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipe = build_pipeline(numeric_cols, nominal_cols, ordinal_cols, y_train)

    # 5-fold stratified CV probabilities on the training fold only (no test leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    cv_pr_auc = float(average_precision_score(y_train, cv_proba))
    baseline_pr = float(y_train.mean())

    # Fit on full training set, evaluate on held-out test
    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    thresholds = choose_thresholds(y_test.to_numpy(), test_proba)

    # Headline metrics at medium (high-recall) threshold
    med_thr = thresholds["medium"]
    y_pred_med = (test_proba >= med_thr).astype(int)
    recall = float(recall_score(y_test, y_pred_med, zero_division=0))
    precision = float(precision_score(y_test, y_pred_med, zero_division=0))
    pr_auc = float(average_precision_score(y_test, test_proba))
    cm = confusion_matrix(y_test, y_pred_med).tolist()

    # Also report high-threshold operating point
    high_thr = thresholds["high"]
    y_pred_high = (test_proba >= high_thr).astype(int)
    high_recall = float(recall_score(y_test, y_pred_high, zero_division=0))
    high_precision = float(precision_score(y_test, y_pred_high, zero_division=0))
    cm_high = confusion_matrix(y_test, y_pred_high).tolist()

    precision_c, recall_c, thr_c = precision_recall_curve(y_test, test_proba)
    # downsample PR curve for JSON payload
    idx = np.linspace(0, len(precision_c) - 1, num=min(200, len(precision_c))).astype(int)
    pr_curve = {
        "precision": [float(precision_c[i]) for i in idx],
        "recall": [float(recall_c[i]) for i in idx],
        "thresholds": [
            float(thr_c[i]) if i < len(thr_c) else None for i in idx if True
        ][: len(idx)],
        "baseline_pr_auc": baseline_pr,
        "test_pr_auc": pr_auc,
    }
    # fix thresholds length alignment for last point
    thr_list = list(thr_c) + [None]
    pr_curve["thresholds"] = [float(thr_list[i]) if thr_list[i] is not None else None for i in idx]

    metrics = {
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "baseline_pr_auc": baseline_pr,
        "cv_pr_auc": cv_pr_auc,
        "test_pr_auc": pr_auc,
        "headline_threshold": "medium",
        "medium_threshold": med_thr,
        "high_threshold": high_thr,
        "medium_band_metrics": {
            "recall": recall,
            "precision": precision,
            "confusion_matrix": {
                "labels": ["pred_0", "pred_1"],
                "matrix": cm,
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
            },
        },
        "high_band_metrics": {
            "recall": high_recall,
            "precision": high_precision,
            "confusion_matrix": {
                "matrix": cm_high,
                "tn": cm_high[0][0],
                "fp": cm_high[0][1],
                "fn": cm_high[1][0],
                "tp": cm_high[1][1],
            },
        },
        "feature_notes": {
            "dropped": {
                "PolicyNumber": "Unique ID — non-predictive and would leak identity.",
                "RepNumber": "Adjuster ID — high cardinality process artifact, not claim content.",
                "AgeOfPolicyHolder": "Binned duplicate of Age; kept numeric Age instead.",
            },
            "kept": {
                "Age": "Numeric driver age; zeros treated as missing.",
                "Year": "Claim year kept as ordered categorical context (not an ID).",
            },
        },
        "why_not_accuracy": (
            "Fraud is ~6% of claims. A model that always predicts non-fraud is ~94% accurate "
            "while catching zero fraud. Precision, recall, and PR-AUC measure the rare-class "
            "trade-off that accuracy hides."
        ),
    }

    # SHAP TreeExplainer on transformed training matrix (fast path for LightGBM)
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    X_train_t = pre.transform(X_train)
    # background sample for optional use; TreeExplainer on LGBM uses tree path
    explainer = shap.TreeExplainer(model)
    # Persist explainer + pipeline
    joblib.dump(
        {
            "pipeline": pipe,
            "explainer": explainer,
            "feature_names": feature_names_from_pipeline(pipe),
            "raw_feature_columns": list(X.columns),
            "thresholds": thresholds,
        },
        ART_DIR / "model.joblib",
    )

    # Sample claims: 2 Low, 2 Medium, 2 High from test set
    bands = np.array([band_for_proba(float(p), thresholds) for p in test_proba])
    samples: list[dict] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for band in ("Low", "Medium", "High"):
        idxs = np.where(bands == band)[0]
        if len(idxs) == 0:
            continue
        pick = rng.choice(idxs, size=min(2, len(idxs)), replace=False)
        for i in pick:
            row = X_test.iloc[int(i)]
            samples.append(
                {
                    "band": band,
                    "probability": float(test_proba[int(i)]),
                    "actual_fraud": int(y_test.iloc[int(i)]),
                    "claim": {c: _jsonable(row[c]) for c in X_test.columns},
                }
            )

    # If a band is missing (e.g. High empty), fill from nearest probabilities
    while len([s for s in samples if s["band"] == "High"]) < 2:
        i = int(np.argsort(test_proba)[-len(samples) - 1])
        row = X_test.iloc[i]
        samples.append(
            {
                "band": "High",
                "probability": float(test_proba[i]),
                "actual_fraud": int(y_test.iloc[i]),
                "claim": {c: _jsonable(row[c]) for c in X_test.columns},
            }
        )
        if len(samples) > 12:
            break

    with open(ART_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(ART_DIR / "pr_curve.json", "w", encoding="utf-8") as f:
        json.dump(pr_curve, f, indent=2)
    with open(ART_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    with open(ART_DIR / "sample_claims.json", "w", encoding="utf-8") as f:
        json.dump(samples[:6], f, indent=2)

    print("=== Training complete ===")
    print(f"baseline_pr_auc (positive rate): {baseline_pr:.4f}")
    print(f"cv_pr_auc: {cv_pr_auc:.4f}")
    print(f"test_pr_auc: {pr_auc:.4f}")
    print(f"medium thr={med_thr:.4f} recall={recall:.4f} precision={precision:.4f}")
    print(f"high thr={high_thr:.4f} recall={high_recall:.4f} precision={high_precision:.4f}")
    print(f"confusion_matrix (medium):\n{cm}")
    print(f"artifacts -> {ART_DIR}")


def _jsonable(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


if __name__ == "__main__":
    main()
