import pickle
from typing import Any, Tuple


def load_pickle(path: str) -> Any:
    """Отварање на Python object од pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_scaler(values, scaler) -> list:
    """
       Скалирање на листа од нумерички features со StandardScaler
    """
    # StandardScaler stores mean_ and scale_ arrays; subtract mean and divide by scale
    return [(val - m) / s if s != 0 else 0.0 for val, m, s in zip(values, scaler.mean_, scaler.scale_)]