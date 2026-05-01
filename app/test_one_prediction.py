from pathlib import Path
from dataclasses import dataclass
import joblib
import pickle
import pandas as pd
import numpy as np


@dataclass
class StackedDeploymentModel:
    preprocessor: object
    base_models: dict
    meta_model: object
    threshold: float
    feature_columns: list

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = PROJECT_ROOT / "artifacts" / "main_model"
SUPPORT_DIR = PROJECT_ROOT / "artifacts" / "support_model"
DATA_PATH = PROJECT_ROOT / "data" / "final_merged_cleaned_preprocessed.csv"


# Load in Artifacts
main = {
    "rf": joblib.load(MAIN_DIR / "rf.pkl"),
    "xgb": joblib.load(MAIN_DIR / "xgb.pkl"),
    "cat": joblib.load(MAIN_DIR / "cat.pkl"),
    "lr": joblib.load(MAIN_DIR / "lr.pkl"),
    "meta": joblib.load(MAIN_DIR / "meta.pkl"),
    "calibrator": joblib.load(MAIN_DIR / "calibrator.pkl"),
    "feature_columns": joblib.load(MAIN_DIR / "feature_columns.pkl"),
    "xgb_shap_model": joblib.load(MAIN_DIR / "xgb_shap_model.pkl"),
    "shap_background": joblib.load(MAIN_DIR / "shap_background.pkl"),
}

with open(MAIN_DIR / "threshold_f1.pkl", "rb") as f:
    main["threshold_f1"] = pickle.load(f)

with open(MAIN_DIR / "threshold_cost.pkl", "rb") as f:
    main["threshold_cost"] = pickle.load(f)

support = {
    "model": joblib.load(SUPPORT_DIR / "stacked_deploy_model.joblib"),
    "feature_columns": joblib.load(SUPPORT_DIR / "feature_columns.joblib"),
    "threshold": joblib.load(SUPPORT_DIR / "selected_threshold.joblib"),
    "threshold_results": pd.read_csv(SUPPORT_DIR / "threshold_results.csv"),
}


# Load in the data
df = pd.read_csv(DATA_PATH)

print("Data shape:", df.shape)

# use 1st row for now
row_idx = 0
row = df.iloc[[row_idx]].copy()

patient_id = row["patientunitstayid"].iloc[0] if "patientunitstayid" in row.columns else "N/A"
print("Testing patientunitstayid:", patient_id)


# helpers
def safe_clip_prob(x):
    x = float(x)
    return max(0.0, min(1.0, x))


def prepare_row(row_df, feature_columns):
    return row_df.reindex(columns=feature_columns, fill_value=0).copy()


def get_positive_proba(model, X):
    probs = model.predict_proba(X)
    return probs[:, 1]


# main model prediction
print("\n Main Model Test")

X_main = prepare_row(row, main["feature_columns"])

rf_prob = get_positive_proba(main["rf"], X_main)[0]
xgb_prob = get_positive_proba(main["xgb"], X_main)[0]
cat_prob = get_positive_proba(main["cat"], X_main)[0]
lr_prob = get_positive_proba(main["lr"], X_main)[0]

base_meta_input_main = np.array([[rf_prob, xgb_prob, cat_prob, lr_prob]])
meta_raw_prob_main = main["meta"].predict_proba(base_meta_input_main)[:, 1][0]

# Apply calibrator if possible
calibrator = main["calibrator"]
if hasattr(calibrator, "predict"):
    main_prob = calibrator.predict(np.array([meta_raw_prob_main]))[0]
else:
    main_prob = meta_raw_prob_main

main_prob = safe_clip_prob(main_prob)

main_pred_f1 = int(main_prob >= main["threshold_f1"])
main_pred_cost = int(main_prob >= main["threshold_cost"])

print(f"rf_prob: {rf_prob:.4f}")
print(f"xgb_prob: {xgb_prob:.4f}")
print(f"cat_prob: {cat_prob:.4f}")
print(f"lr_prob: {lr_prob:.4f}")
print(f"meta_raw_prob: {meta_raw_prob_main:.4f}")
print(f"main_calibrated_prob: {main_prob:.4f}")
print(f"main_pred_f1 @ {main['threshold_f1']:.6f}: {main_pred_f1}")
print(f"main_pred_cost @ {main['threshold_cost']:.6f}: {main_pred_cost}")


# support model prediction
print("\nSupport Model Test")

support_model = support["model"]
X_support_raw = prepare_row(row, support["feature_columns"])

# Preprocess if needed
if support_model.preprocessor is not None:
    X_support = support_model.preprocessor.transform(X_support_raw)
else:
    X_support = X_support_raw

base_probs_support = []
base_model_names = list(support_model.base_models.keys())

for name in base_model_names:
    model = support_model.base_models[name]
    prob = model.predict_proba(X_support)[:, 1][0]
    base_probs_support.append(prob)
    print(f"{name}_prob: {prob:.4f}")

base_meta_input_support = np.array([base_probs_support])
support_prob = support_model.meta_model.predict_proba(base_meta_input_support)[:, 1][0]
support_prob = safe_clip_prob(support_prob)

support_pred = int(support_prob >= support["threshold"])

print(f"support_meta_prob: {support_prob:.4f}")
print(f"support_pred @ {support['threshold']:.6f}: {support_pred}")


# agreement check between models
print("\nAGREEMENT CHECK ")
print("Main (F1 threshold) prediction:", main_pred_f1)
print("Support prediction:", support_pred)

if main_pred_f1 == support_pred:
    print("Models agree.")
else:
    print("Models disagree.")