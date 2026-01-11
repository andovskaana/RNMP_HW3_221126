#!/usr/bin/env python3
"""
Outputs (сите во ./models):
- best_model.pkl
- scaler.pkl
- feature_names.txt
- best_model_info.json  (метрики + избран threshold)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# --------------------------- Data loading ---------------------------

def load_csv_first_col_label(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("CSV must have at least 2 columns (label + >=1 feature).")

        label_name = header[0]
        feature_names = header[1:]

        y_list: List[int] = []
        x_list: List[List[float]] = []

        for row_i, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                y_list.append(int(float(row[0])))
                x_list.append([float(v) for v in row[1:]])
            except Exception as e:
                raise ValueError(f"Failed parsing row {row_i}: {e}")

    X = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(feature_names) != X.shape[1]:
        raise ValueError("Feature name count does not match X columns.")
    return X, y, feature_names

@dataclass
class EvalResult:
    weighted_f1: float
    pos_f1: float
    precision_pos: float
    recall_pos: float
    tp: int
    fp: int
    fn: int
    tn: int
    threshold: float


def eval_with_threshold(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float) -> EvalResult:
    y_pred = (prob_pos >= threshold).astype(int)

    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    pf1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return EvalResult(
        weighted_f1=float(wf1),
        pos_f1=float(pf1),
        precision_pos=float(p),
        recall_pos=float(r),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        threshold=float(threshold),
    )


def find_best_threshold_for_pos_f1(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    """
    Туниран threshold за одлуку за максимизирање на positive-class F1 на валидациски сет, ова деке е imbalanced.
    """
    cand = np.unique(prob_pos)
    if cand.size == 0:
        return 0.5
    if cand.size > 200:
        # sample рамномерно
        idx = np.linspace(0, cand.size - 1, 200).astype(int)
        cand = cand[idx]
    best_t = 0.5
    best_pf1 = -1.0
    for t in cand:
        pf1 = f1_score(y_true, (prob_pos >= t).astype(int), pos_label=1, zero_division=0)
        if pf1 > best_pf1:
            best_pf1 = pf1
            best_t = float(t)
    return float(best_t)

def make_models() -> Dict[str, Any]:

    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "gbt": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def cross_val_score_models(
    X_train_scaled: np.ndarray, y_train: np.ndarray, k: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    CV со default threshold 0.5.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    out: Dict[str, Dict[str, float]] = {}

    for name, model in make_models().items():
        wf1s, pf1s, ps, rs = [], [], [], []

        for tr_idx, va_idx in skf.split(X_train_scaled, y_train):
            Xtr, Xva = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]

            model.fit(Xtr, ytr)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(Xva)[:, 1]
            else:
                # fallback to decision_function (rare here)
                dec = model.decision_function(Xva)
                # map to [0,1] roughly via logistic
                prob = 1.0 / (1.0 + np.exp(-dec))

            res = eval_with_threshold(yva, prob, threshold=0.5)
            wf1s.append(res.weighted_f1)
            pf1s.append(res.pos_f1)
            ps.append(res.precision_pos)
            rs.append(res.recall_pos)

        out[name] = {
            "weighted_f1": float(np.mean(wf1s)),
            "pos_f1": float(np.mean(pf1s)),
            "precision_pos": float(np.mean(ps)),
            "recall_pos": float(np.mean(rs)),
        }

    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("data", "offline.csv"))
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    X, y, feature_names = load_csv_first_col_label(args.csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cross-val споредба на training set
    cv_scores = cross_val_score_models(X_train_scaled, y_train, k=args.cv)
    for name in cv_scores:
        s = cv_scores[name]
        print(f"[CV] Model={name:<18} weighted-F1={s['weighted_f1']:.4f} pos-F1={s['pos_f1']:.4f} "
              f"P={s['precision_pos']:.4f} R={s['recall_pos']:.4f}")

    best_name = max(cv_scores.keys(), key=lambda n: cv_scores[n]["weighted_f1"])
    best_model = make_models()[best_name]

    Xtr, Xva, ytr, yva = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    best_model.fit(Xtr, ytr)
    prob_va = best_model.predict_proba(Xva)[:, 1]
    best_threshold = find_best_threshold_for_pos_f1(yva, prob_va)

    # Refit на цел training set
    best_model.fit(X_train_scaled, y_train)

    # Финслнс евалуација
    prob_test = best_model.predict_proba(X_test_scaled)[:, 1]
    res_default = eval_with_threshold(y_test, prob_test, threshold=0.5)
    res_tuned = eval_with_threshold(y_test, prob_test, threshold=best_threshold)

    print("================ RESULTS ================")
    print(f"Selected best model (by weighted-F1): {best_name}")
    print(f"Test weighted-F1 (thr=0.50): {res_default.weighted_f1:.4f} | pos-F1: {res_default.pos_f1:.4f} "
          f"P: {res_default.precision_pos:.4f} R: {res_default.recall_pos:.4f}")
    print(f"Test weighted-F1 (thr={best_threshold:.3f}): {res_tuned.weighted_f1:.4f} | pos-F1: {res_tuned.pos_f1:.4f} "
          f"P: {res_tuned.precision_pos:.4f} R: {res_tuned.recall_pos:.4f}")
    print(f"Confusion matrix (thr={best_threshold:.3f}): TP={res_tuned.tp} FP={res_tuned.fp} "
          f"FN={res_tuned.fn} TN={res_tuned.tn}")
    print("========================================")

    # Зачувување (pickle) за online фаза
    model_path = os.path.join(args.models_dir, "best_model.pkl")
    scaler_path = os.path.join(args.models_dir, "scaler.pkl")
    features_path = os.path.join(args.models_dir, "feature_names.txt")
    info_path = os.path.join(args.models_dir, "best_model_info.json")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    info = {
        "model_name": best_name,
        "threshold": best_threshold,
        "cv_scores": cv_scores,
        "test_metrics_default_thr_0_5": res_default.__dict__,
        "test_metrics_threshold_tuned": res_tuned.__dict__,
        "class_distribution": {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())},
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Best model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature names saved to {features_path}")
    print(f"Info saved to {info_path}")


if __name__ == "__main__":
    main()
