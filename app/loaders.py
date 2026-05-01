# loaders.py
# loads models, thresholds, shap files, and data with deployment-safe paths

from pathlib import Path
import __main__
import joblib
import pickle
import pandas as pd

from model_defs import StackedDeploymentModel

# helps joblib load the custom support-model class
__main__.StackedDeploymentModel = StackedDeploymentModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = PROJECT_ROOT / "artifacts" / "main_model"
SUPPORT_DIR = PROJECT_ROOT / "artifacts" / "support_model"
DATA_PATH = PROJECT_ROOT / "data" / "final_merged_cleaned_preprocessed.csv"


def _patch_logistic_model(model):
    # fixes sklearn version differences after deployment
    if model is not None and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model


def _safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_main_model():
    main = {}

    # base models
    main["rf"] = joblib.load(MAIN_DIR / "rf.pkl")
    main["xgb"] = joblib.load(MAIN_DIR / "xgb.pkl")
    main["cat"] = joblib.load(MAIN_DIR / "cat.pkl")
    main["lr"] = _patch_logistic_model(joblib.load(MAIN_DIR / "lr.pkl"))

    # meta model + calibration
    main["meta"] = _patch_logistic_model(joblib.load(MAIN_DIR / "meta.pkl"))
    main["calibrator"] = joblib.load(MAIN_DIR / "calibrator.pkl")

    # features
    main["feature_columns"] = list(joblib.load(MAIN_DIR / "feature_columns.pkl"))

    # thresholds
    with open(MAIN_DIR / "threshold_f1.pkl", "rb") as f:
        main["threshold_f1"] = float(pickle.load(f))

    with open(MAIN_DIR / "threshold_cost.pkl", "rb") as f:
        main["threshold_cost"] = float(pickle.load(f))

    # SHAP is optional so it never breaks the app
    main["xgb_shap_model"] = _safe_joblib_load(MAIN_DIR / "xgb_shap_model.pkl")
    main["shap_background"] = _safe_joblib_load(MAIN_DIR / "shap_background.pkl")

    return main


def load_support_model():
    support = {}

    support["model"] = joblib.load(SUPPORT_DIR / "stacked_deploy_model.joblib")
    support["feature_columns"] = list(joblib.load(SUPPORT_DIR / "feature_columns.joblib"))
    support["threshold"] = float(joblib.load(SUPPORT_DIR / "selected_threshold.joblib"))
    support["threshold_results"] = pd.read_csv(SUPPORT_DIR / "threshold_results.csv")

    # patch support model logistic pieces too
    model = support["model"]
    if hasattr(model, "meta_model"):
        model.meta_model = _patch_logistic_model(model.meta_model)
    if hasattr(model, "base_models"):
        for name, m in model.base_models.items():
            model.base_models[name] = _patch_logistic_model(m)

    return support


def load_data():
    return pd.read_csv(DATA_PATH)


def load_all():
    return load_main_model(), load_support_model(), load_data()
