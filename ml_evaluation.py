import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
) 
from typing import Dict
from sklearn.model_selection import train_test_split
from ml_preprocessing import DatasetPreprocessor

def prepare_ml_data(
    df: pd.DataFrame,
    label_attr: str,
    sensitive_attrs: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor: DatasetPreprocessor = None,
) -> dict:
    """
    Preprocess mitigated data and split for ML evaluation.
    
    Uses standard train_test_split with label stratification (common ML practice)
    rather than joint stratification on sensitive attributes. This evaluates
    whether mitigation benefits persist under realistic conditions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Mitigated dataframe.
    label_attr : str
        Name of the target column.
    sensitive_attrs : list[str]
        Sensitive attributes to exclude from features but retain for fairness evaluation.
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.
    preprocessor_kwargs : dict, optional
        Additional arguments for DatasetPreprocessor.
    
    Returns
    -------
    dict with keys:
        X_train, X_test : preprocessed features
        y_train, y_test : labels
        sensitive_train, sensitive_test : sensitive attributes for fairness metrics
        preprocessor : fitted preprocessor
    """
    y = df[label_attr]
    sensitive = df[sensitive_attrs]
    X = df.drop(columns=[label_attr] + sensitive_attrs)
    
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    if preprocessor is None:
        preprocessor = DatasetPreprocessor()
    
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "sensitive_train": sens_train.reset_index(drop=True),
        "sensitive_test": sens_test.reset_index(drop=True),
        "preprocessor": preprocessor
    }


def compute_performance_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute standard ML performance metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = np.nan
    return metrics


def positive_rate(y_pred) -> float:
    """Fraction of positive predictions."""
    return np.mean(y_pred)


def true_positive_rate(y_true, y_pred) -> float:
    """TPR = TP / (TP + FN)."""
    positives = y_true == 1
    if positives.sum() == 0:
        return np.nan
    return y_pred[positives].mean()


def false_positive_rate(y_true, y_pred) -> float:
    """FPR = FP / (FP + TN)."""
    negatives = y_true == 0
    if negatives.sum() == 0:
        return np.nan
    return y_pred[negatives].mean()


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute fairness metrics across sensitive groups.
    
    Returns demographic parity difference and equalized odds difference
    for each sensitive attribute and their intersections.
    """
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Per-attribute fairness
    for col in sensitive.columns:
        groups = sensitive[col].unique()
        
        pos_rates = {}
        tprs = {}
        fprs = {}
        
        for g in groups:
            mask = sensitive[col] == g
            pos_rates[g] = positive_rate(y_pred[mask])
            tprs[g] = true_positive_rate(y_true[mask], y_pred[mask])
            fprs[g] = false_positive_rate(y_true[mask], y_pred[mask])
        
        # Demographic parity: max difference in positive rates
        pr_values = list(pos_rates.values())
        metrics[f"dp_diff_{col}"] = max(pr_values) - min(pr_values)
        
        # Equalized odds: max of TPR diff and FPR diff
        tpr_values = [v for v in tprs.values() if not np.isnan(v)]
        fpr_values = [v for v in fprs.values() if not np.isnan(v)]
        
        tpr_diff = (max(tpr_values) - min(tpr_values)) if len(tpr_values) >= 2 else np.nan
        fpr_diff = (max(fpr_values) - min(fpr_values)) if len(fpr_values) >= 2 else np.nan
        
        metrics[f"eo_diff_{col}"] = max(tpr_diff, fpr_diff) if not (np.isnan(tpr_diff) and np.isnan(fpr_diff)) else np.nan
    
    # Intersectional fairness (if multiple sensitive attributes)
    if len(sensitive.columns) > 1:
        # Create intersectional groups
        sensitive_tuple = sensitive.apply(tuple, axis=1)
        groups = sensitive_tuple.unique()
        
        pos_rates = {g: positive_rate(y_pred[sensitive_tuple == g]) for g in groups}
        pr_values = list(pos_rates.values())
        metrics["dp_diff_intersectional"] = max(pr_values) - min(pr_values)
    
    return metrics


def evaluate_model(model, data: dict) -> Dict[str, float]:
    """Full evaluation: performance + fairness metrics."""
    y_pred = model.predict(data["X_test"])
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(data["X_test"])[:, 1]
    
    metrics = compute_performance_metrics(data["y_test"], y_pred, y_prob)
    
    return metrics