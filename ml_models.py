from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

MODELS = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42), # GBDT
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
}

def get_model(name: str):
    """Return a fresh clone of the model (so it can be refit)."""
    from sklearn.base import clone
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return clone(MODELS[name])