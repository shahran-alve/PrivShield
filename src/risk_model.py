"""Train and load the query-risk classifier.

Trains a logistic regression and a random forest on the query-level features,
selects by held-out macro-F1, and writes both models, the selected model and
its metadata to models/.

Run:  python -m src.risk_model --train
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, GLOBAL_SEED
from src.contracts import RiskLabel, RiskPrediction, ArtifactMissingError
from src.risk_features import FEATURE_ORDER, features_to_row
from src.training_data import TRAINING_CSV
from src.utils import sha256_file

LABELS = ["Safe", "Medium Risk", "High Risk"]

LOGREG_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RF_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
SELECTED_PATH = os.path.join(MODELS_DIR, "selected_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
EVAL_CSV = os.path.join(REPORTS_DIR, "evaluation_results.csv")
CLASSREPORT_CSV = os.path.join(REPORTS_DIR, "classification_report.csv")


def _load_training():
    if not os.path.exists(TRAINING_CSV):
        raise ArtifactMissingError(
            f"{TRAINING_CSV} missing. Run `python -m src.training_data` first.")
    df = pd.read_csv(TRAINING_CSV)
    X = df[FEATURE_ORDER].copy()
    y = df["label"].copy()
    return X, y, df


def _build_models():
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=GLOBAL_SEED)),
    ])
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=GLOBAL_SEED, n_jobs=-1,
    )
    return {"logistic_regression": logreg, "random_forest": rf}


def _metrics(name, model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=LABELS, average="weighted", zero_division=0)

    # High-Risk false-negative rate: true High predicted Safe/Medium.
    high_true = (y_test == "High Risk")
    high_fn = ((y_test == "High Risk") & (y_pred != "High Risk")).sum()
    high_fn_rate = float(high_fn / high_true.sum()) if high_true.sum() else 0.0
    high_recall = float(((y_test == "High Risk") & (y_pred == "High Risk")).sum() /
                        high_true.sum()) if high_true.sum() else 0.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1_macro", n_jobs=-1)

    return {
        "model": name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f_macro), 4),
        "weighted_f1": round(float(f_w), 4),
        "high_risk_recall": round(high_recall, 4),
        "high_risk_fn_rate": round(high_fn_rate, 4),
        "cv_macro_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_macro_f1_std": round(float(cv_scores.std()), 4),
    }


def _plot_class_distribution(y, path):
    counts = y.value_counts().reindex(LABELS).fillna(0)
    plt.figure(figsize=(6, 4))
    plt.bar(LABELS, counts.values, color=["#4c9f70", "#e0a458", "#c0392b"])
    plt.title("Training class distribution")
    plt.ylabel("Episodes")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_confusion(cm, path):
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, int(cm[i, j]), ha="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix (selected model)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_model_comparison(results, path):
    names = [r["model"] for r in results]
    macro = [r["macro_f1"] for r in results]
    hr = [r["high_risk_recall"] for r in results]
    x = np.arange(len(names))
    plt.figure(figsize=(6, 4))
    plt.bar(x - 0.2, macro, 0.4, label="Macro F1")
    plt.bar(x + 0.2, hr, 0.4, label="High-Risk recall")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Model comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    X, y, df = _load_training()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=GLOBAL_SEED)

    models = _build_models()
    fitted, results = {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(_metrics(name, model, X_test, y_test, X_train, y_train))

    # Selection: higher macro-F1, tie (<=0.02) -> Logistic Regression.
    lr = next(r for r in results if r["model"] == "logistic_regression")
    rf = next(r for r in results if r["model"] == "random_forest")
    if rf["macro_f1"] - lr["macro_f1"] > 0.02:
        selected_name, reason = "random_forest", "higher_macro_f1"
    else:
        selected_name = "logistic_regression"
        reason = ("higher_macro_f1" if lr["macro_f1"] >= rf["macro_f1"]
                  else "explainability_tiebreak_within_0.02")

    for r in results:
        r["selected"] = (r["model"] == selected_name)

    joblib.dump(fitted["logistic_regression"], LOGREG_PATH)
    joblib.dump(fitted["random_forest"], RF_PATH)
    joblib.dump(fitted[selected_name], SELECTED_PATH)

    # Metrics + reports.
    pd.DataFrame(results).to_csv(EVAL_CSV, index=False)
    y_pred_sel = fitted[selected_name].predict(X_test)
    rep = classification_report(y_test, y_pred_sel, labels=LABELS,
                                output_dict=True, zero_division=0)
    rep_rows = []
    for key, val in rep.items():
        if isinstance(val, dict):
            rep_rows.append({"row": key, **{k: round(v, 4) for k, v in val.items()}})
        else:
            rep_rows.append({"row": key, "precision": "", "recall": "",
                             "f1-score": round(val, 4), "support": ""})
    pd.DataFrame(rep_rows).to_csv(CLASSREPORT_CSV, index=False)

    # Figures.
    _plot_class_distribution(y, os.path.join(FIGURES_DIR, "class_distribution.png"))
    cm = confusion_matrix(y_test, y_pred_sel, labels=LABELS)
    _plot_confusion(cm, os.path.join(FIGURES_DIR, "confusion_matrix.png"))
    _plot_model_comparison(results, os.path.join(FIGURES_DIR, "model_comparison.png"))

    # Explainability tables.
    coef_table = _logreg_coefficients(fitted["logistic_regression"])
    importance_table = _rf_importances(fitted["random_forest"])

    metadata = {
        "feature_order": FEATURE_ORDER,
        "labels": LABELS,
        "selected_model": selected_name,
        "selection_reason": reason,
        "metrics": results,
        "training_csv_sha256": sha256_file(TRAINING_CSV),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "logreg_coefficients": coef_table,
        "rf_importances": importance_table,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[risk_model] selected {selected_name} ({reason})")
    for r in results:
        print(f"  {r['model']}: macroF1={r['macro_f1']} "
              f"HiRecall={r['high_risk_recall']} HiFNrate={r['high_risk_fn_rate']}")
    return metadata


def _logreg_coefficients(pipeline):
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_
    table = {}
    for ci, cls in enumerate(clf.classes_):
        table[str(cls)] = {FEATURE_ORDER[i]: round(float(coefs[ci][i]), 4)
                           for i in range(len(FEATURE_ORDER))}
    return table


def _rf_importances(rf):
    return {FEATURE_ORDER[i]: round(float(rf.feature_importances_[i]), 4)
            for i in range(len(FEATURE_ORDER))}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
_CACHE = {}


def load_selected_model():
    if "model" not in _CACHE:
        if not os.path.exists(SELECTED_PATH):
            raise ArtifactMissingError(
                "selected_model.joblib missing. Run `python -m src.risk_model --train`.")
        _CACHE["model"] = joblib.load(SELECTED_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["feature_order"] != FEATURE_ORDER:
            raise ArtifactMissingError("Model feature order does not match code.")
        _CACHE["meta"] = meta
    return _CACHE["model"], _CACHE["meta"]


def predict_risk(features: dict) -> RiskPrediction:
    model, meta = load_selected_model()
    row = features_to_row(features)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    prob_map = {c: float(proba[i]) for i, c in enumerate(classes)}
    for lab in LABELS:
        prob_map.setdefault(lab, 0.0)

    label = max(prob_map, key=prob_map.get)
    risk_score = prob_map["High Risk"] + 0.5 * prob_map["Medium Risk"]
    risk_score = round(min(1.0, max(0.0, risk_score)), 4)

    top = _top_factors(meta, features, label)
    return RiskPrediction(
        label=RiskLabel(label),
        risk_score=risk_score,
        probabilities={k: round(v, 4) for k, v in prob_map.items()},
        top_factors=top,
    )


def _top_factors(meta, features, label):
    pretty = {
        "log_min_result_group_size": "small minimum group size",
        "uses_sensitive_field": "a sensitive field was requested",
        "sensitivity_score": "high field sensitivity",
        "sequence_narrowing_score": "narrowing across recent queries",
        "query_repetition_score": "repeated similar queries",
        "filter_specificity_score": "a very specific filter",
        "num_filters": "several filters combined",
    }
    if meta.get("selected_model") == "logistic_regression":
        weights = meta.get("logreg_coefficients", {}).get(label, {})
        contrib = {f: weights.get(f, 0.0) * (features.get(f, 0.0) or 0.0)
                   for f in FEATURE_ORDER}
        ranked = sorted(contrib, key=lambda f: abs(contrib[f]), reverse=True)
    else:
        imp = meta.get("rf_importances", {})
        ranked = sorted(FEATURE_ORDER, key=lambda f: imp.get(f, 0.0), reverse=True)
    factors = []
    for f in ranked:
        if f in pretty and pretty[f] not in factors:
            factors.append(pretty[f])
        if len(factors) >= 3:
            break
    return factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    train_models()
