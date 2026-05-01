# ui_helpers.py

# This file keeps UI wording, color logic, display names,
# and small interpretation helpers in one place.


# Risk label + color
def get_risk_category(prob):
    # Low / Moderate / High bands
    if prob < 0.30:
        return "Low Risk", "#4CAF50"
    elif prob < 0.70:
        return "Moderate Risk", "#F2B84B"
    else:
        return "High Risk", "#E35D5D"


# Recommendation text
def get_recommendation(prob):
    # Short clinical action guidance
    if prob < 0.30:
        return "Standard monitoring"
    elif prob < 0.70:
        return "Close observation recommended"
    else:
        return "Immediate ICU attention"


# Agreement message between models
def get_agreement_message(agree):
    if agree:
        return "✅ Models agree"
    return "⚠️ Models disagree — review recommended"


# Sidebar trust text
def get_confidence_text():
    return [
        "Prediction based on first 24h ICU data",
        "Probability output is calibrated",
        "Uses second model for additional validation"
    ]


# Sidebar limitations text
def get_limitations():
    return [
        "Depends on input data quality",
        "Trained on historical ICU data",
        "May not generalize to all patient populations",
        "Not a substitute for clinical judgment"
    ]


# Cleaner feature display names
FEATURE_NAME_MAP = {
    "age": "Age",
    "bmi": "BMI",
    "admissionheight": "Admission Height",
    "admissionweight": "Admission Weight",
    "respiratoryrate": "Respiratory Rate",
    "heartrate": "Heart Rate",
    "heart_rate": "Heart Rate",
    "wbc": "White Blood Cell Count",
    "sodium": "Sodium",
    "bun": "BUN",
    "bun_mean": "BUN",
    "creatinine": "Creatinine",
    "creatinine_mean": "Creatinine",
    "intaketotal": "Total Intake",
    "pred_missing": "Prediction Missingness Flag",
    "numbedscategory": "Number of Beds Category",
    "hx_heme": "Hematologic History",
    "hx_none": "No Major Prior History Indicator",
    "sao2_min": "Minimum SpO2",
    "sao2_mean": "SpO2",
    "map_mean": "Mean Arterial Pressure",
    "glucose_mean": "Glucose",
    "rr_mean": "Respiratory Rate",
    "hr_mean": "Heart Rate",
    "temp_mean": "Temperature",
    "dialysis": "Dialysis",
    "dbp_min": "DBP Min",
    "dbp": "DBP",
    "bilirubin": "Bilirubin",
    "ph": "pH"
}


# Short clinician-facing meaning for common features
FEATURE_MEANING_MAP = {
    "respiratoryrate": "Higher breathing rate can reflect respiratory distress or physiologic stress.",
    "heartrate": "Higher heart rate can reflect stress, compensation, or instability.",
    "heart_rate": "Higher heart rate can reflect stress, compensation, or instability.",
    "wbc": "Abnormal white blood cell levels may reflect infection or systemic inflammation.",
    "sodium": "Abnormal sodium may reflect fluid or metabolic imbalance.",
    "bun": "Elevated BUN can suggest kidney dysfunction, dehydration, or critical illness burden.",
    "bun_mean": "Elevated BUN can suggest kidney dysfunction, dehydration, or critical illness burden.",
    "creatinine": "Elevated creatinine can suggest reduced kidney function.",
    "creatinine_mean": "Elevated creatinine can suggest reduced kidney function.",
    "intaketotal": "Fluid intake can reflect treatment intensity and physiologic support needs.",
    "pred_missing": "Missingness itself may signal incomplete information or care complexity in the record.",
    "numbedscategory": "Bed category may indirectly reflect unit size or care setting differences.",
    "hx_heme": "Hematologic history may reflect underlying disease burden.",
    "hx_none": "This may represent absence of a major prior history category in the encoded data.",
    "sao2_min": "Lower oxygen saturation can indicate respiratory compromise.",
    "sao2_mean": "Lower oxygen saturation can indicate respiratory compromise.",
    "map_mean": "Lower blood pressure may reflect hemodynamic instability.",
    "glucose_mean": "Abnormal glucose may reflect metabolic stress or illness severity.",
    "rr_mean": "Higher breathing rate can reflect respiratory distress or physiologic stress.",
    "hr_mean": "Higher heart rate can reflect stress, compensation, or instability.",
    "temp_mean": "Temperature extremes may reflect infection or systemic stress.",
    "age": "Older age is often associated with greater clinical vulnerability.",
    "bmi": "BMI may influence physiologic reserve and comorbidity burden.",
    "dialysis": "Dialysis can reflect severe renal disease or high illness complexity.",
    "dbp_min": "Very low diastolic blood pressure can reflect poor perfusion or instability.",
    "dbp": "Low diastolic blood pressure can reflect poor perfusion or instability.",
    "bilirubin": "Abnormal bilirubin can reflect liver dysfunction or systemic illness burden.",
    "ph": "Abnormal pH can reflect acid-base imbalance and physiologic instability."
}


# Convert raw column name to cleaner display text
def pretty_feature_name(name):
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    return str(name).replace("_", " ").title()


# Keep numeric display to 2 decimals
def format_feature_value(value):
    try:
        value = float(value)
        return f"{value:.2f}"
    except Exception:
        return str(value)


# Turn raw direction into cleaner wording
def pretty_effect_label(direction):
    if direction == "Increase Risk":
        return "Raises Risk"
    return "Lowers Risk"


# Short meaning for table
def get_clinical_meaning(feature):
    if feature in FEATURE_MEANING_MAP:
        return FEATURE_MEANING_MAP[feature]
    return "This feature influenced the model's estimate for this patient."


# Short bullet explanation for the top-driver section
def explain_driver(feature, value, direction):
    clean_name = pretty_feature_name(feature)
    clean_value = format_feature_value(value)
    effect_label = pretty_effect_label(direction).lower()

    return f"{clean_name} ({clean_value}) {effect_label} in this estimate."


# Support model explanation text
def get_support_model_explainer():
    return (
        "The support model acts as a second opinion. Its main role is to validate or question "
        "the primary model's estimate, especially in more uncertain cases."
    )


# Agreement explanation text
def get_agreement_explainer(agree):
    if agree:
        return (
            "Both models reached the same overall decision, which strengthens confidence "
            "in this assessment."
        )
    return (
        "The models differ, which may reflect uncertainty or a borderline case. "
        "This is a signal for closer clinical review."
    )


# Convert support-model internal spread into clinician-friendly language
def get_support_consensus_level(base_probs):
    values = list(base_probs.values())

    if len(values) == 0:
        return "Unknown"

    spread = max(values) - min(values)

    if spread <= 0.10:
        return "High"
    elif spread <= 0.25:
        return "Moderate"
    return "Mixed"


# Clinician-friendly support-model insight
def get_support_model_insight(main_prob, support_prob, base_probs):
    consensus = get_support_consensus_level(base_probs)

    diff = support_prob - main_prob

    if diff <= -0.10:
        direction_text = "The second opinion estimates lower risk than the primary model."
    elif diff >= 0.10:
        direction_text = "The second opinion estimates higher risk than the primary model."
    else:
        direction_text = "The second opinion is broadly aligned with the primary model."

    if consensus == "High":
        consensus_text = "Signals inside the support model were relatively consistent."
    elif consensus == "Moderate":
        consensus_text = "Signals inside the support model showed moderate consistency."
    else:
        consensus_text = "Signals inside the support model were mixed, which may indicate a more borderline case."

    return f"{direction_text} {consensus_text}"


# Explain threshold role in plain language
def get_threshold_role_text(kind):
    if kind == "f1":
        return "Balanced cutoff for overall precision-recall tradeoff."
    if kind == "cost":
        return "Safety-oriented cutoff that flags more potential risk."
    if kind == "support":
        return "Second-opinion cutoff used for validation or review."
    return "Decision threshold used in deployment."