# predictors.py
# runs the main model and support model safely for one patient row

import numpy as np
import pandas as pd


def safe_clip_prob(x):
    x = float(np.asarray(x).ravel()[0])
    return max(0.0, min(1.0, x))


def prepare_row(row_df, feature_columns):
    # align to exactly the columns the model was trained on
    feature_columns = [str(c) for c in list(feature_columns)]
    X = row_df.reindex(columns=feature_columns, fill_value=0).copy()

    # models need numeric input only
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def get_positive_proba(model, X):
    probs = model.predict_proba(X)
    return np.asarray(probs)[:, 1]


def predict_main(main, row):
    X = prepare_row(row, main["feature_columns"])

    rf_prob = get_positive_proba(main["rf"], X)[0]
    xgb_prob = get_positive_proba(main["xgb"], X)[0]
    cat_prob = get_positive_proba(main["cat"], X)[0]
    lr_prob = get_positive_proba(main["lr"], X)[0]

    meta_input = np.array([[rf_prob, xgb_prob, cat_prob, lr_prob]])
    meta_raw = main["meta"].predict_proba(meta_input)[:, 1][0]

    calibrator = main.get("calibrator")
    if calibrator is not None and hasattr(calibrator, "predict"):
        prob = calibrator.predict(np.array([meta_raw]))[0]
    else:
        prob = meta_raw

    prob = safe_clip_prob(prob)
    threshold_f1 = float(main["threshold_f1"])
    threshold_cost = float(main["threshold_cost"])

    return {
        "prob": prob,
        "pred_f1": int(prob >= threshold_f1),
        "pred_cost": int(prob >= threshold_cost),
        "threshold_f1": threshold_f1,
        "threshold_cost": threshold_cost,
        "base_probs": {
            "rf": safe_clip_prob(rf_prob),
            "xgb": safe_clip_prob(xgb_prob),
            "cat": safe_clip_prob(cat_prob),
            "lr": safe_clip_prob(lr_prob),
        },
    }


def predict_support(support, row):
    
    model = support["model"]

    # STEP 1: force correct columns
    X = row.reindex(columns=support["feature_columns"], fill_value=0)

    # STEP 2: ensure DataFrame (NOT numpy)
    X = pd.DataFrame(X, columns=support["feature_columns"])

    # STEP 3: apply preprocessor safely
    if model.preprocessor is not None:
        X_processed = model.preprocessor.transform(X)
    else:
        X_processed = X

    # STEP 4: base model probs
    base_probs = []
    for name, m in model.base_models.items():
        prob = m.predict_proba(X_processed)[:, 1][0]
        base_probs.append(prob)

    # STEP 5: meta model
    meta_input = np.array([base_probs])
    prob = model.meta_model.predict_proba(meta_input)[:, 1][0]

    return {
        "prob": float(prob),
        "pred": int(prob >= support["threshold"]),
        "base_probs": dict(zip(model.base_models.keys(), base_probs))
    }


def compare_predictions(main_result, support_result):
    main_pred = main_result["pred_f1"]
    support_pred = support_result["pred"]
    return {
        "main_pred": main_pred,
        "support_pred": support_pred,
        "agree": main_pred == support_pred,
    }
