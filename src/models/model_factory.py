"""Model factory: instantiate any supported model from a name + config dict."""
from __future__ import annotations

from .base_model import BaseModel

SUPPORTED_MODELS = ("arima", "svr", "random_forest", "lightgbm", "lstm", "gru", "tcn")

# How fold_runner should preprocess inputs for each model.
MODEL_KIND = {
    "tcn": "nn",
    "lstm": "nn",
    "gru": "nn",
    "svr": "sklearn",
    "random_forest": "sklearn",
    "lightgbm": "sklearn",
    "arima": "arima",
}


def model_kind(name: str) -> str:
    """Return ``'nn'`` | ``'sklearn'`` | ``'arima'`` for a model name."""
    key = name.lower()
    if key not in MODEL_KIND:
        raise ValueError(f"Unknown model name: {name!r}. Supported: {', '.join(SUPPORTED_MODELS)}")
    return MODEL_KIND[key]


def build_model(name: str, cfg: dict | None = None) -> BaseModel:
    """Return an instantiated model. ``name`` is case-insensitive."""
    cfg = dict(cfg or {})
    key = name.lower()

    if key == "arima":
        from .arima_model import ARIMAModel

        return ARIMAModel(**cfg)
    if key == "svr":
        from .svr_model import SVRModel

        return SVRModel(**cfg)
    if key == "random_forest":
        from .random_forest_model import RandomForestModel

        return RandomForestModel(**cfg)
    if key == "lightgbm":
        from .lightgbm_model import LightGBMModel

        return LightGBMModel(**cfg)
    if key == "lstm":
        from .lstm_model import LSTMModel

        return LSTMModel(**cfg)
    if key == "gru":
        from .gru_model import GRUModel

        return GRUModel(**cfg)
    if key == "tcn":
        from .tcn_model import TCNModel

        return TCNModel(**cfg)

    raise ValueError(f"Unknown model name: {name!r}. Supported: {', '.join(SUPPORTED_MODELS)}")
