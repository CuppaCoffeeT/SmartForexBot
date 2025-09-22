# model.py
from sklearn.ensemble import RandomForestClassifier

def get_model(params=None):
    """
    Initialize a RandomForestClassifier with optional params.
    Args:
        params: dict of hyperparameters
    Returns:
        RandomForestClassifier
    """
    default_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    if params:
        default_params.update(params)
    return RandomForestClassifier(**default_params)