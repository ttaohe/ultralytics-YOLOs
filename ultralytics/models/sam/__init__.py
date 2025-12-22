# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__all__ = (
    "SAM",
    "Predictor",
    "SAM2Predictor",
    "SAM2VideoPredictor",
    "SAM2DynamicInteractivePredictor",
)


def __getattr__(name):
    if name == "SAM":
        from .model import SAM

        return SAM
    if name in {"Predictor", "SAM2Predictor", "SAM2VideoPredictor", "SAM2DynamicInteractivePredictor"}:
        from . import predict as _predict

        return getattr(_predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
