"""Модуль расчета метрик качества бинарной классификации и агрегации результатов."""

import os
import json
from typing import Dict, List
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """
    Вычисляет метрики классификации:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    Параметр zero_division=0 предотвращает исключения при делении на ноль.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def update_summary_csv(
    results_dir: str,
    exp_name: str,
    model_short: str,
    metrics: Dict[str, float],
) -> None:
    """Обновляет или создает summary CSV файл для конкретного типа эксперимента."""
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{exp_name}_summary_metrics.csv")

    schema = {
        "Model": str,
        "Accuracy": float,
        "Precision": float,
        "Recall": float,
        "F1-Score": float,
    }

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path).astype(schema)
        except Exception:
            df = pd.DataFrame(columns=list(schema.keys())).astype(schema)
    else:
        df = pd.DataFrame(columns=list(schema.keys())).astype(schema)

    m_name = model_short.replace("/", "_").replace("-", "_")
    acc = metrics["accuracy"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1_score"]

    if m_name in df["Model"].values:
        df.loc[df["Model"] == m_name, ["Accuracy", "Precision", "Recall", "F1-Score"]] = [
            acc,
            prec,
            rec,
            f1,
        ]
    else:
        new_row = pd.DataFrame(
            [
                {
                    "Model": str(m_name),
                    "Accuracy": float(acc),
                    "Precision": float(prec),
                    "Recall": float(rec),
                    "F1-Score": float(f1),
                }
            ]
        ).astype(schema)
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_path, index=False)
