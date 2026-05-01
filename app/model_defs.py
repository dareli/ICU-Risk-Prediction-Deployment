# custom objects used when loading saved models

from dataclasses import dataclass

@dataclass
class StackedDeploymentModel:
    preprocessor: object # how data is transformed
    base_models: dict # rf, xgb, cat, lr, svm
    meta_model: object # final decision model
    threshold: float # decision cutoff
    feature_columns: list # expected input columns