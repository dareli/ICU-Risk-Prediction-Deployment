
# streamlit_app.py

import io
import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px

from loaders import load_all
from predictors import predict_main, predict_support, compare_predictions
from ui_helpers import (
    get_risk_category,
    get_recommendation,
    get_agreement_message,
    get_confidence_text,
    get_limitations,
    pretty_feature_name,
    get_support_model_explainer,
    get_agreement_explainer,
    get_clinical_meaning,
    pretty_effect_label,
    format_feature_value,
    get_support_model_insight,
    get_support_consensus_level,
    get_threshold_role_text,
)
from explainers import get_top_shap_features, get_brief_driver_summary


st.set_page_config(page_title="ICU Risk Alert System", layout="wide")


# Styling
st.markdown("""
<style>
    .main {
        background-color: #F5F7F7;
    }

    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.00rem;
    }

    .title-text {
        font-size: 2.35rem;
        font-weight: 800;
        color: #0B5E6E;
        margin-bottom: 0.10rem;
        line-height: 1.15;
    }

    .subtitle-text {
        font-size: 1.00rem;
        color: #5A6C74;
        margin-bottom: 1.00rem;
    }

    .section-header {
        font-size: 1.18rem;
        font-weight: 800;
        color: #0B5E6E;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
    }

    .card {
        background-color: white;
        padding: 1.10rem 1.20rem;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.00rem;
        border: 1px solid #E7ECEF;
    }

    .big-risk {
        font-size: 2.45rem;
        font-weight: 800;
        color: #0B5E6E;
        margin-bottom: 0.20rem;
    }

    .small-label {
        font-size: 0.92rem;
        color: #60737B;
        margin-bottom: 0.20rem;
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        color: white;
        font-weight: 700;
        font-size: 0.90rem;
        margin-top: 0.15rem;
        margin-bottom: 0.55rem;
    }

    .recommendation {
        font-size: 1.00rem;
        font-weight: 700;
        color: #17313B;
        margin-top: 0.35rem;
    }

    .mini-note {
        color: #60737B;
        font-size: 0.90rem;
        margin-top: 0.35rem;
    }

    .info-box {
        background-color: #F2FAFA;
        border: 1px solid #D8EEEE;
        border-radius: 12px;
        padding: 0.80rem 1.00rem;
        margin-top: 0.50rem;
        margin-bottom: 0.50rem;
        color: #134B57;
    }

    .summary-card {
        background-color: white;
        border: 1px solid #E7ECEF;
        border-radius: 14px;
        padding: 1.00rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        text-align: left;
        min-height: 118px;
    }

    .summary-label {
        color: #5D6F77;
        font-size: 0.90rem;
        margin-bottom: 0.20rem;
    }

    .summary-value {
        color: #0B5E6E;
        font-size: 1.90rem;
        font-weight: 800;
    }

    .summary-sub {
        color: #74858D;
        font-size: 0.85rem;
        margin-top: 0.15rem;
    }

    .soft-note {
        color: #60737B;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# Loaders
@st.cache_resource
def get_loaded_objects():
    return load_all()


main, support, df = get_loaded_objects()


HIDDEN_DRIVER_FEATURES = {"pred_missing"}


@st.cache_data(show_spinner=False)
def get_cached_top_shap_features(patient_id, row_json):
    row_df = pd.read_json(io.StringIO(row_json), orient="split")

    try:
        shap_df = get_top_shap_features(main, row_df, top_n=10)
        if shap_df is None:
            return None
        shap_df = shap_df.copy()
        shap_df = shap_df[~shap_df["feature"].isin(HIDDEN_DRIVER_FEATURES)].copy()
        shap_df = shap_df.head(5).reset_index(drop=True)
        return shap_df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def score_loaded_cohort(row_count):
    # cached off row count; internally uses currently loaded df/models
    if "bad_outcome" not in df.columns:
        return None

    main_probs = []
    support_probs = []
    agree_flags = []

    for i in range(len(df)):
        row = df.iloc[[i]].copy()
        main_result = predict_main(main, row)
        support_result = predict_support(support, row)
        comparison = compare_predictions(main_result, support_result)

        main_probs.append(float(main_result["prob"]))
        support_probs.append(float(support_result["prob"]))
        agree_flags.append(int(comparison["agree"]))

    high_rate = sum(p >= 0.70 for p in main_probs) / len(main_probs)
    moderate_rate = sum((p >= 0.30 and p < 0.70) for p in main_probs) / len(main_probs)
    low_rate = sum(p < 0.30 for p in main_probs) / len(main_probs)
    agreement_rate = sum(agree_flags) / len(agree_flags)

    avg_main_risk = sum(main_probs) / len(main_probs)
    avg_support_risk = sum(support_probs) / len(support_probs)

    return {
        "main_probs": main_probs,
        "support_probs": support_probs,
        "agree_flags": agree_flags,
        "high_rate": high_rate,
        "moderate_rate": moderate_rate,
        "low_rate": low_rate,
        "agreement_rate": agreement_rate,
        "avg_main_risk": avg_main_risk,
        "avg_support_risk": avg_support_risk,
    }


try:
    cohort_metrics = score_loaded_cohort(len(df))
except Exception:
    cohort_metrics = None


# Helpers
def first_existing_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def safe_numeric(series_value, default=0.0):
    try:
        value = float(series_value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def bounded_default(value, lo, hi):
    v = safe_numeric(value, default=(lo + hi) / 2)
    v = min(max(v, lo), hi)
    return v


def get_patient_options(dataframe):
    if "patientunitstayid" in dataframe.columns:
        return dataframe["patientunitstayid"].tolist(), "patientunitstayid"
    return list(range(len(dataframe))), None


def add_summary_card(col, label, value, sub):
    with col:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
                <div class="summary-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# Sidebar
st.sidebar.markdown("## 🩺 About this tool")
st.sidebar.write(
    "This dashboard is an ICU decision-support prototype designed to identify "
    "early bad outcome risk using first-24-hour patient data."
)

st.sidebar.markdown("## ✅ Confidence & Reliability")
for line in get_confidence_text():
    st.sidebar.write(f"- {line}")

st.sidebar.markdown("## ⚠️ Limitations")
for line in get_limitations():
    st.sidebar.write(f"- {line}")


# Header
st.markdown('<div class="title-text">ICU Risk Alert System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Balancing resource allocation and patient safety</div>',
    unsafe_allow_html=True
)


patient_tab, user_input_tab, cohort_tab, performance_tab = st.tabs(
    ["🧍 Patient View", "✍️ User Input", "📊 Cohort Summary", "📈 Monitoring"]
)


# PATIENT TAB
with patient_tab:
    options, id_col = get_patient_options(df)

    if id_col is not None:
        selected_patient_id = st.selectbox("Select Patient ID", options)
        selected_row = df[df[id_col] == selected_patient_id].iloc[[0]].copy()
    else:
        selected_patient_id = st.selectbox("Select Patient Row", options)
        selected_row = df.iloc[[selected_patient_id]].copy()

    main_result = predict_main(main, selected_row)
    main_prob = main_result["prob"]
    risk_label, risk_color = get_risk_category(main_prob)
    main_recommendation = get_recommendation(main_prob)

    col1, col2 = st.columns([1, 1.30])

    with col1:
        st.markdown('<div class="section-header">🧍 Patient Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.write(f"**Patient ID:** {selected_patient_id}")

        overview_fields = [
            "age", "bmi", "gender", "ethnicity", "unittype",
            "admissionheight", "admissionweight"
        ]
        for col in overview_fields:
            if col in selected_row.columns:
                raw_value = selected_row.iloc[0][col]
                clean_value = format_feature_value(raw_value)
                st.write(f"**{pretty_feature_name(col)}:** {clean_value}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**🔎 Why trust this prediction?**")
        st.write("- Based on first 24h ICU data")
        st.write("- Uses a calibrated primary probability")
        st.write("- Includes a second model for validation")
        st.write("- Intended to support, not replace, clinician judgment")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🚨 Primary Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<div class="small-label">Estimated bad outcome risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-risk">{main_prob * 100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="pill" style="background-color: {risk_color};">{risk_label}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="recommendation">Recommended Action: {main_recommendation}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="mini-note">Primary model uses the calibrated F1-based decision threshold.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="info-box">ℹ️ This estimate is meant to support clinical decision-making, not replace clinician judgment.</div>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Why this patient is at risk</div>', unsafe_allow_html=True)

    row_json = selected_row.to_json(orient="split")
    shap_df = get_cached_top_shap_features(str(selected_patient_id), row_json)

    if shap_df is None or shap_df.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info(
            "Patient-level explanation is temporarily unavailable in the deployed app. "
            "The risk prediction and second-opinion model are still available."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        shap_df = shap_df.copy()
        shap_df["display_feature"] = shap_df["feature"].apply(pretty_feature_name)

        left, right = st.columns([1.35, 1])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            plot_df = shap_df.sort_values("shap_value", ascending=True).copy()
            bar_colors = ["#E35D5D" if val >= 0 else "#2B7BB9" for val in plot_df["shap_value"]]

            fig, ax = plt.subplots(figsize=(8.20, 4.80))
            ax.barh(plot_df["display_feature"], plot_df["shap_value"], color=bar_colors)
            ax.axvline(0, color="#9AA8AE", linewidth=1)
            ax.set_xlabel("Impact on Risk Estimate")
            ax.set_ylabel("Feature")
            ax.set_title("Top factors influencing this risk estimate")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Red bars increase risk. Blue bars decrease risk.")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**🧾 Key contributing factors**")

            display_df = shap_df.copy()
            display_df["Effect on Risk"] = display_df["direction"].apply(pretty_effect_label)
            display_df["Clinical Meaning"] = display_df["feature"].apply(get_clinical_meaning)
            display_df["Feature"] = display_df["display_feature"]
            display_df = display_df[["Feature", "Effect on Risk", "Clinical Meaning"]]

            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.write("**What this means**")
            st.write(
                "These are the main features that most influenced the model's estimate for this patient. "
                "The effect label shows whether each feature pushed the estimate upward or downward."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**💡 Explanation**")
        top_rows = shap_df.head(3).copy()
        for _, row in top_rows.iterrows():
            feature_name = pretty_feature_name(row["feature"])
            effect_label = pretty_effect_label(row["direction"]).lower()
            st.write(f"- {feature_name} {effect_label} in this estimate.")
        st.caption(get_brief_driver_summary(shap_df))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🛡️ Model Second Opinion</div>', unsafe_allow_html=True)
    show_second_opinion = st.toggle("Get Second Opinion")

    if show_second_opinion:
        support_result = predict_support(support, selected_row)
        comparison = compare_predictions(main_result, support_result)

        support_prob = support_result["prob"]
        support_label, support_color = get_risk_category(support_prob)
        support_recommendation = get_recommendation(support_prob)
        agreement_message = get_agreement_message(comparison["agree"])

        support_insight = get_support_model_insight(
            main_prob, support_prob, support_result["base_probs"]
        )
        support_consensus = get_support_consensus_level(support_result["base_probs"])

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Support Model Risk**")
            st.markdown(f'<div class="big-risk">{support_prob * 100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pill" style="background-color: {support_color};">{support_label}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="recommendation">Recommended Action: {support_recommendation}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="mini-note">{get_support_model_explainer()}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="info-box">🧠 Why the second opinion may differ: {support_insight}</div>',
                unsafe_allow_html=True
            )
            st.write(f"**Internal support-model consensus:** {support_consensus}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**🤝 Model Agreement**")
            st.write(agreement_message)
            diff = abs(main_prob - support_prob) * 100
            st.write(f"**Probability difference:** {diff:.2f}%")
            if comparison["agree"]:
                st.success(get_agreement_explainer(True))
            else:
                st.warning(get_agreement_explainer(False))
            st.markdown("</div>", unsafe_allow_html=True)

    # What-if demo
    st.markdown('<div class="section-header">🧪 Optional What-If Demo</div>', unsafe_allow_html=True)

    available_sim_features = [
        ("Heart Rate", first_existing_column(selected_row.columns, ["hr_mean", "heartrate", "heart_rate"]), 50, 150),
        ("APACHE Score", first_existing_column(selected_row.columns, ["apache_score", "apachescore"]), 0, 120),
        ("MAP", first_existing_column(selected_row.columns, ["map_mean"]), 40, 120),
        ("SpO2", first_existing_column(selected_row.columns, ["sao2_mean", "sao2_min"]), 70, 100),
        ("Creatinine", first_existing_column(selected_row.columns, ["creatinine_mean", "creatinine"]), 0, 8),
        ("Respiratory Rate", first_existing_column(selected_row.columns, ["rr_mean", "respiratoryrate"]), 5, 45),
        ("Temperature", first_existing_column(selected_row.columns, ["temp_mean"]), 93, 106),
    ]
    available_sim_features = [item for item in available_sim_features if item[1] is not None]

    with st.expander("Try a lightweight patient simulation"):
        if len(available_sim_features) == 0:
            st.info("No matching simulation features were found in the loaded dataset.")
        else:
            st.markdown(
                '<div class="soft-note">This is a presentation demo only. It shows how the estimated risk could respond when common ICU features change.</div>',
                unsafe_allow_html=True
            )

            sim_row = selected_row.copy()
            slider_cols = st.columns(min(3, len(available_sim_features)))

            for idx, (label, col_name, lo, hi) in enumerate(available_sim_features[:6]):
                with slider_cols[idx % len(slider_cols)]:
                    default_val = bounded_default(selected_row.iloc[0][col_name], lo, hi)

                    # integer-style sliders for presentation simplicity
                    if label in {"Heart Rate", "APACHE Score", "MAP", "SpO2", "Respiratory Rate"}:
                        slider_val = st.slider(label, int(lo), int(hi), int(round(default_val)))
                    else:
                        slider_val = st.slider(label, float(lo), float(hi), float(round(default_val, 2)))

                    sim_row[col_name] = slider_val

            sim_result = predict_main(main, sim_row)
            sim_prob = sim_result["prob"]
            sim_label, sim_color = get_risk_category(sim_prob)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Simulated primary risk:** {sim_prob * 100:.2f}%")

            st.markdown(
                f'<div class="pill" style="background-color: {sim_color};">{sim_label}</div>',
                unsafe_allow_html=True
            )
            st.write(
                "Use this as a presentation demo for how the system could react to changing patient conditions."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📘 Model Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(
        "This prototype is intended to support clinical decision-making, not replace clinician judgment. "
        "Predictions are based on early ICU data and should be interpreted alongside the patient's full clinical picture."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# USER INPUT TAB
with user_input_tab:
    st.markdown('<div class="section-header">✍️ Manual ICU Patient Input</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(
        "Enter custom patient values to test how the ICU risk model responds outside of the existing loaded cohort."
    )
    st.warning(
        "This is still a prototype. Predictions are based on patterns learned from the eICU demo dataset and should not be used as a clinical decision tool."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Enter Patient Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 65)
        apache = st.number_input("APACHE Score", 0, 120, 50)
        hr = st.number_input("Heart Rate (mean)", 40, 180, 90)
        map_val = st.number_input("MAP", 40, 130, 75)

    with col2:
        rr = st.number_input("Respiratory Rate", 5, 50, 18)
        spo2 = st.number_input("SpO2 (%)", 70, 100, 96)
        temp = st.number_input("Temperature (°F)", 93.0, 106.0, 98.6)
        glucose = st.number_input("Glucose", 40, 400, 120)

    with col3:
        creatinine = st.number_input("Creatinine", 0.1, 10.0, 1.0)
        bun = st.number_input("BUN", 1, 100, 14)
        albumin = st.number_input("Albumin", 1.0, 5.0, 3.5)

    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Other"])
    unittype = st.selectbox("ICU Unit Type", ["MICU", "SICU", "CCU", "Other"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Build one custom patient row
    custom_row = pd.DataFrame([{
        "age": age,
        "apachescore": apache,
        "hr_mean": hr,
        "map_mean": map_val,
        "rr_mean": rr,
        "sao2_mean": spo2,
        "temp_mean": temp,
        "glucose_mean": glucose,
        "creatinine_mean": creatinine,
        "bun_mean": bun,
        "albumin": albumin,
        "gender": gender,
        "ethnicity": ethnicity,
        "unittype": unittype,
    }])

    if st.button("Run Custom Patient Prediction"):
        custom_main_result = predict_main(main, custom_row)
        custom_support_result = predict_support(support, custom_row)
        custom_comparison = compare_predictions(custom_main_result, custom_support_result)

        custom_main_prob = custom_main_result["prob"]
        custom_support_prob = custom_support_result["prob"]

        custom_risk_label, custom_risk_color = get_risk_category(custom_main_prob)
        custom_recommendation = get_recommendation(custom_main_prob)

        st.markdown('<div class="section-header">🚨 Custom Patient Risk Assessment</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Primary Model Risk**")
            st.markdown(f'<div class="big-risk">{custom_main_prob * 100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pill" style="background-color: {custom_risk_color};">{custom_risk_label}</div>',
                unsafe_allow_html=True
            )
            st.write(f"**Recommended Action:** {custom_recommendation}")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            support_label, support_color = get_risk_category(custom_support_prob)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Support Model Risk**")
            st.markdown(f'<div class="big-risk">{custom_support_prob * 100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pill" style="background-color: {support_color};">{support_label}</div>',
                unsafe_allow_html=True
            )

            if custom_comparison["agree"]:
                st.success("✅ Models agree")
            else:
                st.warning("⚠️ Models disagree — review recommended")

            diff = abs(custom_main_prob - custom_support_prob) * 100
            st.write(f"**Probability difference:** {diff:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

    
        # Interactive SHAP explanation for custom patient
        st.markdown('<div class="section-header">📌 Custom Patient Risk Drivers</div>', unsafe_allow_html=True)

        try:
            custom_shap_df = get_top_shap_features(main, custom_row, top_n=10)
        except Exception:
            custom_shap_df = None

        if custom_shap_df is None or custom_shap_df.empty:
            st.info("Custom SHAP explanation is temporarily unavailable in the deployed app.")
        else:
            custom_shap_df = custom_shap_df[~custom_shap_df["feature"].isin(HIDDEN_DRIVER_FEATURES)].copy()
            custom_shap_df = custom_shap_df.head(8).reset_index(drop=True)

            custom_shap_df["Feature"] = custom_shap_df["feature"].apply(pretty_feature_name)
            custom_shap_df["Effect"] = custom_shap_df["direction"].apply(pretty_effect_label)
            custom_shap_df["SHAP Impact"] = custom_shap_df["shap_value"]

            plot_df = custom_shap_df.sort_values("SHAP Impact", ascending=True)

            fig_custom_shap = px.bar(
                plot_df,
                x="SHAP Impact",
                y="Feature",
                orientation="h",
                color="Effect",
                hover_data={
                    "Feature": True,
                    "Effect": True,
                    "SHAP Impact": ":.4f"
                },
                title="Top factors influencing the custom risk estimate"
            )

            fig_custom_shap.add_vline(x=0, line_width=1, line_dash="dash")
            fig_custom_shap.update_layout(
                height=430,
                xaxis_title="Impact on Risk Estimate",
                yaxis_title="Feature",
                legend_title="Effect"
            )

            st.plotly_chart(fig_custom_shap, use_container_width=True)
            st.caption("Positive SHAP values push the risk estimate higher. Negative SHAP values push it lower.")

# COHORT SUMMARY TAB
with cohort_tab:
    st.markdown('<div class="section-header">📊 ICU Cohort Summary</div>', unsafe_allow_html=True)
    st.caption("High-level cohort snapshot for the currently loaded ICU population")

    # Basic cohort counts
    total_patients = len(df)

    if "bad_outcome" in df.columns:
        observed_bad = int(df["bad_outcome"].sum())
        observed_bad_rate = observed_bad / total_patients
        observed_good = total_patients - observed_bad
    else:
        observed_bad = 0
        observed_bad_rate = 0.00
        observed_good = total_patients

    if cohort_metrics is not None:
        high_risk_rate = cohort_metrics["high_rate"]
        moderate_risk_rate = cohort_metrics["moderate_rate"]
        low_risk_rate = cohort_metrics["low_rate"]
    else:
        high_risk_rate = 0.00
        moderate_risk_rate = 0.00
        low_risk_rate = 0.00

    # Top summary cards
    c1, c2, c3 = st.columns(3)
    add_summary_card(c1, "👥 Total Patients", f"{total_patients:,}", "Current loaded cohort")
    add_summary_card(c2, "🚨 Observed Bad Outcomes", f"{observed_bad:,}", "Historical positive cases in loaded data")
    add_summary_card(c3, "📈 Observed Bad Outcome Rate", f"{observed_bad_rate * 100:.2f}%", "Loaded cohort event rate")

    st.write("")
    r1, r2, r3 = st.columns(3)
    add_summary_card(r1, "🔴 High-Risk Flag Rate", f"{high_risk_rate * 100:.2f}%", "Patients flagged high risk by primary model")
    add_summary_card(r2, "🟡 Moderate-Risk Share", f"{moderate_risk_rate * 100:.2f}%", "Patients in the middle risk band")
    add_summary_card(r3, "🟢 Low-Risk Share", f"{low_risk_rate * 100:.2f}%", "Patients in the low-risk band")

    # Helper for subgroup charts
    # Works with raw columns like ethnicity/unittype OR one-hot encoded columns like ethnicity_White
    def build_subgroup_rate_table(dataframe, raw_col, dummy_prefixes):
        if "bad_outcome" not in dataframe.columns:
            return pd.DataFrame(columns=["Group", "Bad Outcome Rate", "Patient Count"])

        if raw_col in dataframe.columns:
            out = (
                dataframe.groupby(raw_col, dropna=False)["bad_outcome"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={raw_col: "Group", "mean": "Bad Outcome Rate", "count": "Patient Count"})
            )
            out["Group"] = out["Group"].astype(str)
            return out.sort_values("Bad Outcome Rate", ascending=False)

        dummy_cols = []
        for prefix in dummy_prefixes:
            dummy_cols.extend([c for c in dataframe.columns if str(c).startswith(prefix)])

        rows = []
        for col in sorted(set(dummy_cols)):
            mask = dataframe[col] == 1
            n = int(mask.sum())
            if n > 0:
                clean_group = str(col)
                for prefix in dummy_prefixes:
                    clean_group = clean_group.replace(prefix, "")
                clean_group = clean_group.replace("_", " ").strip().title()
                rows.append({
                    "Group": clean_group,
                    "Bad Outcome Rate": float(dataframe.loc[mask, "bad_outcome"].mean()),
                    "Patient Count": n,
                })

        return pd.DataFrame(rows).sort_values("Bad Outcome Rate", ascending=False) if rows else pd.DataFrame(columns=["Group", "Bad Outcome Rate", "Patient Count"])

    left_col, right_col = st.columns([1.12, 1])

    # Left side: charts only
    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**📉 Observed cohort outcomes**")

        outcome_df = pd.DataFrame({
            "Outcome": ["No bad outcome", "Bad outcome"],
            "Count": [observed_good, observed_bad]
        })

        fig2, ax2 = plt.subplots(figsize=(6.50, 4.00))
        ax2.bar(outcome_df["Outcome"], outcome_df["Count"], color=["#1FA3A3", "#E35D5D"])
        ax2.set_ylabel("Patient Count")
        ax2.set_title("Observed Outcome Distribution")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

        if cohort_metrics is not None:
            band_counts = pd.Series({
                "Low Risk": int(round(cohort_metrics["low_rate"] * total_patients)),
                "Moderate Risk": int(round(cohort_metrics["moderate_rate"] * total_patients)),
                "High Risk": int(round(cohort_metrics["high_rate"] * total_patients)),
            })

            fig3, ax3 = plt.subplots(figsize=(6.50, 4.00))
            ax3.bar(band_counts.index, band_counts.values, color=["#4CAF50", "#F2B84B", "#E35D5D"])
            ax3.set_ylabel("Patient Count")
            ax3.set_title("Predicted Risk Band Distribution")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

        # Subgroup chart: ICU type
        unit_df = build_subgroup_rate_table(
            df,
            raw_col="unittype",
            dummy_prefixes=["unittype_", "unit_type_", "icu_type_", "icu_unit_type_"]
        ).head(8)

        if not unit_df.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**🏥 Bad outcome rate by ICU unit type**")

            fig4, ax4 = plt.subplots(figsize=(6.50, 4.00))
            ax4.barh(unit_df["Group"].astype(str), unit_df["Bad Outcome Rate"] * 100, color="#2B7BB9")
            ax4.set_xlabel("Bad Outcome Rate (%)")
            ax4.set_title("Observed Rate by Unit Type")
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)

            st.caption("This helps show whether observed outcomes differ across ICU settings.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Subgroup chart: ethnicity
        eth_df = build_subgroup_rate_table(
            df,
            raw_col="ethnicity",
            dummy_prefixes=["ethnicity_", "ethnic_", "race_", "race_ethnicity_"]
        )

        if not eth_df.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**🧬 Bad outcome rate by ethnicity**")

            fig_eth, ax_eth = plt.subplots(figsize=(6.50, 4.00))
            ax_eth.barh(eth_df["Group"].astype(str), eth_df["Bad Outcome Rate"] * 100, color="#6C8EBF")
            ax_eth.set_xlabel("Bad Outcome Rate (%)")
            ax_eth.set_title("Observed Rate by Ethnicity")
            ax_eth.spines["top"].set_visible(False)
            ax_eth.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_eth)

            st.caption("Dataset contains a dominant group; results may reflect underlying population imbalance.")
            st.markdown("</div>", unsafe_allow_html=True)

    # Right side: interpretation only
    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**🧠 How to use this panel**")
        st.write(
            "- This tab summarizes the loaded ICU cohort at a population level.\n"
            "- Observed outcome cards show historical event burden.\n"
            "- Risk-share cards show how the model distributes patients across low, moderate, and high risk.\n"
            "- Subgroup charts help analysts check where dataset imbalance may affect interpretation."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**💡 Key Takeaways**")

        observed_rate = observed_bad_rate * 100
        high_flag_rate = high_risk_rate * 100

        if high_flag_rate > observed_rate:
            behavior = "more conservative (flags more patients as high risk)"
        else:
            behavior = "more selective (flags fewer patients as high risk)"

        st.write(
            f"- Observed bad outcomes occur in **{observed_rate:.2f}%** of patients.\n"
            f"- The model flags approximately **{high_flag_rate:.2f}%** of patients as high risk.\n"
            f"- This suggests the model is **{behavior}**.\n"
            f"- This balance can be adjusted depending on clinical priorities (e.g., catching more risk vs reducing alert fatigue)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**⚖️ Subgroup caution**")
        st.write(
            "The subgroup charts are not a full fairness audit. They are included as a quick deployment check, "
            "especially because the demo dataset may contain uneven representation across ethnicity and ICU type."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# MONITORING TAB
with performance_tab:
    st.markdown('<div class="section-header">📈 Model Performance & Monitoring</div>', unsafe_allow_html=True)
    st.caption("Threshold roles, agreement behavior, and deployment monitoring signals")

    t1, t2, t3 = st.columns(3)
    add_summary_card(t1, "🎯 Balanced Threshold", f"{main['threshold_f1']:.2f}", get_threshold_role_text("f1"))
    add_summary_card(t2, "🛟 Safety Threshold", f"{main['threshold_cost']:.2f}", get_threshold_role_text("cost"))
    add_summary_card(t3, "🛡️ Second-Opinion Threshold", f"{support['threshold']:.2f}", get_threshold_role_text("support"))

    st.write("")

    if cohort_metrics is not None:
        a1, a2, a3 = st.columns(3)
        add_summary_card(a1, "🤝 Agreement Rate", f"{cohort_metrics['agreement_rate'] * 100:.2f}%", "How often both models reached the same decision")
        add_summary_card(a2, "🧠 Avg Primary Risk", f"{cohort_metrics['avg_main_risk'] * 100:.2f}%", "Average primary-model estimated risk")
        add_summary_card(a3, "🛡️ Avg Support Risk", f"{cohort_metrics['avg_support_risk'] * 100:.2f}%", "Average support-model estimated risk")

        st.write("")
        left_info, right_info = st.columns([1, 1])

        with left_info:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**📊 Agreement monitoring**")

            agree_n = int(sum(cohort_metrics["agree_flags"]))
            disagree_n = int(len(cohort_metrics["agree_flags"]) - agree_n)

            fig5, ax5 = plt.subplots(figsize=(6.50, 4.00))
            ax5.bar(["Agree", "Disagree"], [agree_n, disagree_n], color=["#1FA3A3", "#E35D5D"])
            ax5.set_ylabel("Patient Count")
            ax5.set_title("Primary vs Support Model Agreement")
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5)
            st.markdown("</div>", unsafe_allow_html=True)

        with right_info:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**🧭 Risk profile comparison**")

            fig6, ax6 = plt.subplots(figsize=(6.50, 4.00))
            ax6.hist(cohort_metrics["main_probs"], bins=20, alpha=0.65, label="Primary Model", color="#2B7BB9")
            ax6.hist(cohort_metrics["support_probs"], bins=20, alpha=0.45, label="Support Model", color="#F2B84B")
            ax6.set_xlabel("Predicted Risk")
            ax6.set_ylabel("Patient Count")
            ax6.set_title("Distribution of Estimated Risk")
            ax6.legend()
            ax6.spines["top"].set_visible(False)
            ax6.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig6)
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    info_left, info_right = st.columns([1, 1])

    with info_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**🧾 What clinicians may use here**")
        st.write("- The current decision thresholds")
        st.write("- How often the second opinion agrees")
        st.write("- Whether the overall risk profile looks stable or shifted")
        st.markdown("</div>", unsafe_allow_html=True)

    with info_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**🧠 What analysts may use here**")
        st.write("- Threshold review before deployment updates")
        st.write("- Agreement vs disagreement volume")
        st.write("- Average risk drift between the primary and support models")
        st.write("- A base panel that can later connect to user feedback logs")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**⚠️ Interpreting Model Disagreement**")

    st.write(
        "- Cases where the primary and support models disagree may represent borderline or uncertain patients.\n"
        "- These cases are useful signals for clinician review rather than immediate action.\n"
        "- Over time, disagreement patterns can help guide model improvement and threshold tuning."
    )

    st.markdown("</div>", unsafe_allow_html=True) 
