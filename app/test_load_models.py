from pathlib import Path
from dataclasses import dataclass
import joblib
import pickle
import pandas as pd

@dataclass
class StackedDeploymentModel:
    preprocessor: object
    base_models: dict
    meta_model: object
    threshold: float
    feature_columns: list


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = PROJECT_ROOT / "artifacts" / "main_model"
SUPPORT_DIR = PROJECT_ROOT / "artifacts" / "support_model"
DATA_PATH = PROJECT_ROOT / "data" / "final_merged_cleaned_preprocessed.csv"

print("PATH CHECK")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("MAIN_DIR exists:", MAIN_DIR.exists())
print("SUPPORT_DIR exists:", SUPPORT_DIR.exists())
print("DATA_PATH exists:", DATA_PATH.exists())

print("\nLOAD MAIN MODEL ARTIFACTS:")
main_artifacts = {}

main_artifacts["rf"] = joblib.load(MAIN_DIR / "rf.pkl")
print("Loaded: rf.pkl")

main_artifacts["xgb"] = joblib.load(MAIN_DIR / "xgb.pkl")
print("Loaded: xgb.pkl")

main_artifacts["cat"] = joblib.load(MAIN_DIR / "cat.pkl")
print("Loaded: cat.pkl")

main_artifacts["lr"] = joblib.load(MAIN_DIR / "lr.pkl")
print("Loaded: lr.pkl")

main_artifacts["meta"] = joblib.load(MAIN_DIR / "meta.pkl")
print("Loaded: meta.pkl")

main_artifacts["calibrator"] = joblib.load(MAIN_DIR / "calibrator.pkl")
print("Loaded: calibrator.pkl")

main_artifacts["feature_columns"] = joblib.load(MAIN_DIR / "feature_columns.pkl")
print("Loaded: feature_columns.pkl")

with open(MAIN_DIR / "threshold_f1.pkl", "rb") as f:
    main_artifacts["threshold_f1"] = pickle.load(f)
print("Loaded: threshold_f1.pkl")

with open(MAIN_DIR / "threshold_cost.pkl", "rb") as f:
    main_artifacts["threshold_cost"] = pickle.load(f)
print("Loaded: threshold_cost.pkl")

main_artifacts["xgb_shap_model"] = joblib.load(MAIN_DIR / "xgb_shap_model.pkl")
print("Loaded: xgb_shap_model.pkl")

main_artifacts["shap_background"] = joblib.load(MAIN_DIR / "shap_background.pkl")
print("Loaded: shap_background.pkl")


print("\nLOAD SUPPORT MODEL ARTIFACTS")
support_artifacts = {}

support_artifacts["model"] = joblib.load(SUPPORT_DIR / "stacked_deploy_model.joblib")
print("Loaded: stacked_deploy_model.joblib")

support_artifacts["feature_columns"] = joblib.load(SUPPORT_DIR / "feature_columns.joblib")
print("Loaded: feature_columns.joblib")

support_artifacts["threshold"] = joblib.load(SUPPORT_DIR / "selected_threshold.joblib")
print("Loaded: selected_threshold.joblib")

support_artifacts["threshold_results"] = pd.read_csv(SUPPORT_DIR / "threshold_results.csv")
print("Loaded: threshold_results.csv")


print("\nLOAD DATA")
df = pd.read_csv(DATA_PATH)
print("Data shape:", df.shape)
print("First 10 columns:", df.columns[:10].tolist())

print("\nFEATURE SUMMARY")
print("Main feature count:", len(main_artifacts["feature_columns"]))
print("Support feature count:", len(support_artifacts["feature_columns"]))

print("\nSUPPORT MODEL OBJECT SUMMARY")
print("Support model type:", type(support_artifacts["model"]))

if hasattr(support_artifacts["model"], "base_models"):
    print("Support base models:", list(support_artifacts["model"].base_models.keys()))

if hasattr(support_artifacts["model"], "feature_columns"):
    print("Support model embedded feature count:", len(support_artifacts["model"].feature_columns))

print("\nTHRESHOLDS")
print("Main threshold_f1:", main_artifacts["threshold_f1"])
print("Main threshold_cost:", main_artifacts["threshold_cost"])
print("Support threshold:", support_artifacts["threshold"])

print("\nall loading checks completed.")