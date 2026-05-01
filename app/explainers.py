# explainers.py
# SHAP/local explanation helpers. SHAP is optional so deployment never crashes.

import numpy as np
import pandas as pd
import shap


def prepare_shap_row(row_df, feature_columns):
    feature_columns = [str(c) for c in list(feature_columns)]
    X = row_df.reindex(columns=feature_columns, fill_value=0).copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


def get_shap_values(main, row_df):
    if main.get("xgb_shap_model") is None or main.get("shap_background") is None:
        return None, None

    try:
        X_shap = prepare_shap_row(row_df, main["feature_columns"])
        explainer = shap.Explainer(
            main["xgb_shap_model"].predict,
            main["shap_background"],
        )
        shap_values = explainer(X_shap)
        return shap_values, X_shap
    except Exception:
        return None, None


def get_top_shap_features(main, row_df, top_n=5):
    shap_values, X_shap = get_shap_values(main, row_df)

    if shap_values is None or X_shap is None:
        return None

    vals = np.asarray(shap_values.values[0]).ravel()
    feats = X_shap.iloc[0]

    out = pd.DataFrame({
        "feature": X_shap.columns,
        "feature_value": feats.values,
        "shap_value": vals,
    })

    out["abs_shap_value"] = out["shap_value"].abs()
    out["direction"] = out["shap_value"].apply(
        lambda x: "Increase Risk" if x > 0 else "Decrease Risk"
    )

    return out.sort_values("abs_shap_value", ascending=False).head(top_n).reset_index(drop=True)


def get_brief_driver_summary(shap_df):
    if shap_df is None or shap_df.empty:
        return "SHAP drivers are unavailable for this deployed run."

    features = shap_df["feature"].tolist()

    if len(features) == 1:
        return f"Top driver included {features[0]}."
    if len(features) == 2:
        return f"Top drivers included {features[0]} and {features[1]}."
    return f"Top drivers included {features[0]}, {features[1]}, and {features[2]}."
