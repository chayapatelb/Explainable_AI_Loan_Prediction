# -*- coding: utf-8 -*-
"""
Fair AI Dashboard (Streamlit)
=============================
Run the training script first (it creates the ./models folder), then:

    streamlit run fair_ai_dashboard.py
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import shap
import streamlit as st
from fairlearn.metrics import demographic_parity_difference
from sklearn.metrics import accuracy_score

try:
    from codecarbon import EmissionsTracker
except Exception:  # codecarbon is optional
    EmissionsTracker = None

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="Fair AI Dashboard", page_icon="🤖", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SEED = 42

REQUIRED_FILES = [
    "fair_model.pkl", "baseline_model.pkl", "scaler.pkl", "feature_names.pkl",
    "X_test_scaled.pkl", "y_test.pkl", "sens_test.pkl", "shap_background.pkl",
]

st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #1E88E5; text-align: center; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 1rem; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fairness-good { color: #00a050; font-weight: bold; }
    .fairness-bad { color: #ff4444; font-weight: bold; }
    .explanation-box {
        background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;
        border-left: 5px solid #1E88E5; margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def positive_class_shap(values):
    """2-D (samples x features) SHAP array for the positive class, for any shap version."""
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def fair_positive_proba(model, X):
    """P(income > 50K) from a fairlearn ExponentiatedGradient model.
    (ExponentiatedGradient has no predict_proba; its weighted vote is _pmf_predict.)"""
    X = np.asarray(X)
    if hasattr(model, "_pmf_predict"):
        return np.asarray(model._pmf_predict(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {MODELS_DIR}: {', '.join(missing)}. "
            "Run fair_ai_prediction.py first."
        )
    return {
        "fair_model": joblib.load(MODELS_DIR / "fair_model.pkl"),
        "baseline_model": joblib.load(MODELS_DIR / "baseline_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / "scaler.pkl"),
        "feature_names": joblib.load(MODELS_DIR / "feature_names.pkl"),
        "X_test_scaled": joblib.load(MODELS_DIR / "X_test_scaled.pkl"),
        "y_test": joblib.load(MODELS_DIR / "y_test.pkl"),
        "sens_test": joblib.load(MODELS_DIR / "sens_test.pkl"),
        "background": joblib.load(MODELS_DIR / "shap_background.pkl"),
    }


@st.cache_resource(show_spinner="Preparing SHAP explainers...")
def load_explainers():
    art = load_artifacts()
    baseline_explainer = shap.TreeExplainer(art["baseline_model"])
    # The fair model is an ensemble of classifiers, not a tree model, so use the
    # model-agnostic KernelExplainer on its positive-class probability.
    fair_model = art["fair_model"]
    fair_explainer = shap.KernelExplainer(
        lambda X: fair_positive_proba(fair_model, X),
        shap.kmeans(art["background"], 10),
    )
    return {"baseline": baseline_explainer, "fair": fair_explainer}


@st.cache_resource(show_spinner="Evaluating models on the test set...")
def compute_comparison():
    art = load_artifacts()
    X, y, sens = art["X_test_scaled"], art["y_test"], art["sens_test"]
    fair_pred = art["fair_model"].predict(X, random_state=SEED)
    base_pred = art["baseline_model"].predict(X)
    return pd.DataFrame({
        "Model": ["Fair Model", "Baseline"],
        "Accuracy": [accuracy_score(y, fair_pred), accuracy_score(y, base_pred)],
        "Fairness Gap": [
            demographic_parity_difference(y, fair_pred, sensitive_features=sens),
            demographic_parity_difference(y, base_pred, sensitive_features=sens),
        ],
    })


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown("<h1 class='main-header'>🎯 Fair AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Model Explainability + Energy Tracking + Fairness Metrics")

try:
    art = load_artifacts()
except Exception as exc:
    st.error(f"Error loading models: {exc}")
    st.stop()

with st.sidebar:
    st.title("⚙️ Controls")
    model_choice = st.radio(
        "Select Model",
        ["Fair Model (Demographic Parity)", "Baseline Model (Unconstrained)"],
        help="Choose which model to use for predictions",
    )
    use_fair = model_choice.startswith("Fair")

    st.subheader("📝 Input Features")
    age = st.slider("Age", 18, 80, 35)
    education = st.slider("Education Years", 1, 16, 13)
    hours = st.slider("Hours/Week", 10, 80, 40)
    gender = st.selectbox("Gender", ["Female", "Male"])
    race = st.selectbox("Race", ["Non-White", "White"])
    gender_num = 1 if gender == "Male" else 0
    race_num = 1 if race == "White" else 0

    st.subheader("🔬 Analysis Options")
    show_shap = st.checkbox("Show SHAP Analysis", True)
    track_energy = st.checkbox("Track Energy Consumption", True)

    predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Type", "Fair Model" if use_fair else "Baseline",
              delta="⚖️ Fairness Constrained" if use_fair else "📈 Max Accuracy")
with col2:
    st.metric("Features Used", "5", delta="Age, Edu, Hours, Gender, Race")
with col3:
    st.metric("Protected Attribute", "Gender")
with col4:
    st.metric("Fairness Metric", "Demographic Parity")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if predict_btn:
    raw_features = np.array([[age, education, hours, gender_num, race_num]])
    features_scaled = art["scaler"].transform(raw_features)

    tracker = None
    if track_energy and EmissionsTracker is not None:
        try:
            tracker = EmissionsTracker(save_to_file=False, log_level="error")
            tracker.start()
        except Exception:
            tracker = None

    cpu_before = psutil.cpu_percent(interval=0.1)
    mem_before = psutil.virtual_memory().percent
    t0 = time.perf_counter()

    if use_fair:
        probability = float(fair_positive_proba(art["fair_model"], features_scaled)[0])
    else:
        probability = float(art["baseline_model"].predict_proba(features_scaled)[0][1])
    prediction = int(probability >= 0.5)

    inference_time = time.perf_counter() - t0
    cpu_usage = (cpu_before + psutil.cpu_percent(interval=None)) / 2
    memory_usage = (mem_before + psutil.virtual_memory().percent) / 2

    emissions = 0.0
    if tracker is not None:
        try:
            value = tracker.stop()
            emissions = float(value) if value is not None else 0.0
        except Exception:
            emissions = 0.0

    st.markdown("---")
    st.header("📊 Prediction Results")
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        if prediction == 1:
            st.markdown("<div class='metric-card'><h2>💰 HIGH INCOME</h2>"
                        "<p>&gt;50K/year</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, "
                        "#FF6B6B 0%, #FF8E53 100%);'><h2>📉 LOW INCOME</h2>"
                        "<p>&lt;=50K/year</p></div>", unsafe_allow_html=True)
    with r2:
        st.metric("Probability of >50K", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    with r3:
        if track_energy:
            st.metric("Inference Time", f"{inference_time * 1000:.2f} ms")
            st.metric("CO₂ Emissions", f"{emissions:.6f} kg")
    with r4:
        if gender == "Female":
            if prediction == 1:
                status, css = "⚖️ Positive outcome for the protected group", "fairness-good"
            else:
                status, css = "⚠️ Check for potential bias", "fairness-bad"
        else:
            status, css = "✅ Privileged-group prediction", ""
        st.markdown("**Fairness Check:**")
        st.markdown(f"<p class='{css}'>{status}</p>", unsafe_allow_html=True)

    # ---- SHAP ---------------------------------------------------------
    if show_shap:
        st.markdown("---")
        st.header("🧠 SHAP Explainability Analysis")
        with st.spinner("Generating SHAP explanations..."):
            explainers = load_explainers()
            if use_fair:
                raw_shap = explainers["fair"].shap_values(features_scaled, nsamples=200, silent=True)
            else:
                raw_shap = explainers["baseline"].shap_values(features_scaled)
            shap_row = positive_class_shap(raw_shap)[0]

            shap_df = pd.DataFrame({
                "Feature": art["feature_names"],
                "SHAP Value": shap_row,
                "Feature Value": [age, education, hours, gender_num, race_num],
            })
            shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()
            shap_df = shap_df.sort_values("Absolute SHAP", ascending=True)

            s1, s2 = st.columns([2, 1])
            with s1:
                fig = go.Figure(go.Bar(
                    y=shap_df["Feature"], x=shap_df["SHAP Value"], orientation="h",
                    marker_color=["#00cc96" if v > 0 else "#ef553b" for v in shap_df["SHAP Value"]],
                    text=shap_df["SHAP Value"].round(3), textposition="outside",
                ))
                fig.update_layout(title="Feature Contributions to Prediction",
                                  xaxis_title="SHAP value (impact on model output)",
                                  yaxis_title="Feature", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with s2:
                top = shap_df.iloc[-1]
                st.markdown(
                    "<div class='explanation-box'>"
                    "<b>📌 Interpretation</b><br>"
                    "• Positive SHAP (green) → raises the chance of HIGH income<br>"
                    "• Negative SHAP (red) → lowers it<br>"
                    "• Larger bars → more influence on this prediction<br><br>"
                    f"<b>🔍 Top contributor:</b> {top['Feature']} (value: {top['Feature Value']}), "
                    f"contribution {abs(top['SHAP Value']):.3f}"
                    "</div>", unsafe_allow_html=True)

            with st.expander("📋 View Detailed SHAP Values"):
                st.dataframe(
                    shap_df[["Feature", "Feature Value", "SHAP Value", "Absolute SHAP"]]
                    .style.format({"SHAP Value": "{:.4f}", "Absolute SHAP": "{:.4f}"})
                )

    # ---- Energy -------------------------------------------------------
    if track_energy:
        st.markdown("---")
        st.header("⚡ Energy & Resource Tracking")
        e1, e2, e3 = st.columns(3)
        gauges = [
            (e1, inference_time * 1000, "Inference Time (ms)", 50, "#1E88E5"),
            (e2, cpu_usage, "CPU Usage (%)", 100, "#00cc96"),
            (e3, memory_usage, "Memory Usage (%)", 100, "#FF6B6B"),
        ]
        for column, value, title, upper, color in gauges:
            with column:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=value, title={"text": title},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={"axis": {"range": [0, upper]}, "bar": {"color": color}},
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"🌍 Carbon footprint: {emissions:.6f} kg CO₂ for this prediction")
        st.caption(f"≈ {emissions * 1000:.4f} grams of CO₂")

# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------
st.markdown("---")
st.header("📈 Model Performance Comparison")

comparison_df = compute_comparison()
fair_gap = float(comparison_df.loc[0, "Fairness Gap"])
baseline_gap = float(comparison_df.loc[1, "Fairness Gap"])

c1, c2 = st.columns(2)
colors = {"Fair Model": "#1E88E5", "Baseline": "#FF6B6B"}
with c1:
    fig = px.bar(comparison_df, x="Model", y="Accuracy", title="Accuracy Comparison",
                 color="Model", color_discrete_map=colors)
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.bar(comparison_df, x="Model", y="Fairness Gap",
                 title="Fairness Gap Comparison (lower is better)",
                 color="Model", color_discrete_map=colors)
    fig.update_layout(showlegend=False, yaxis_range=[0, max(0.2, baseline_gap * 1.1)])
    st.plotly_chart(fig, use_container_width=True)

improvement = baseline_gap - fair_gap
st.metric(
    "Fairness Improvement", f"{improvement:.3f}",
    delta=(f"{improvement / baseline_gap * 100:.1f}% reduction in disparity"
           if baseline_gap > 0 else None),
)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>"
    "Built with ❤️ using Streamlit | Fair AI Dashboard v2.1<br>"
    "SHAP explainability · energy tracking · fairness metrics</div>",
    unsafe_allow_html=True,
)
