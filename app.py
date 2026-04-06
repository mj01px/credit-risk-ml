"""Streamlit app for credit risk prediction."""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "models/best_model.pkl"

st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()
    pipeline = artifact["pipeline"]
    model_name = artifact["model_name"]
    metrics = artifact["metrics"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Credit Risk")
    st.caption("AI-powered credit default predictor")
    st.divider()

    st.subheader("Applicant Profile")

    age = st.slider("Age", 18, 75, 35)
    annual_income = st.number_input(
        "Annual Income ($)", min_value=12_000, max_value=200_000,
        value=55_000, step=1_000, format="%d"
    )
    employment_years = st.slider("Years Employed", 0, 40, 5)

    st.subheader("Loan Details")

    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=1_000, max_value=60_000,
        value=15_000, step=500, format="%d"
    )
    loan_duration_months = st.selectbox(
        "Duration (months)", [12, 24, 36, 48, 60], index=2
    )
    loan_purpose = st.selectbox(
        "Purpose",
        ["personal", "car", "education", "home_improvement", "medical"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    st.subheader("Credit Profile")

    credit_history = st.selectbox(
        "Credit History", ["good", "fair", "poor"],
        format_func=str.title
    )
    num_credit_lines = st.slider("Number of Credit Lines", 0, 15, 3)

    predict_btn = st.button("Analyze Risk", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Credit Risk Analyzer")
st.caption("Machine learning model to predict the probability of loan default.")

if not MODEL_LOADED:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

# Model info
with st.expander("Model Info", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_name)
    col2.metric("Test AUC", metrics.get("test_auc", "—"))
    col3.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")

st.divider()

if predict_btn:
    dti = (loan_amount / loan_duration_months * 12) / annual_income

    input_df = pd.DataFrame([{
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_duration_months": loan_duration_months,
        "num_credit_lines": num_credit_lines,
        "debt_to_income_ratio": round(dti, 3),
        "loan_purpose": loan_purpose,
        "credit_history": credit_history,
    }])

    proba = pipeline.predict_proba(input_df)[0, 1]
    prediction = int(proba >= 0.5)

    # ── Result card ────────────────────────────────────────────────────────────
    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if prediction == 1:
            st.error("### HIGH RISK — Likely to Default")
        else:
            st.success("### LOW RISK — Unlikely to Default")

        st.metric("Default Probability", f"{proba:.1%}")
        st.metric("Debt-to-Income Ratio", f"{dti:.1%}")

        risk_label = (
            "Very Low" if proba < 0.25
            else "Low" if proba < 0.40
            else "Medium" if proba < 0.55
            else "High" if proba < 0.75
            else "Very High"
        )
        st.info(f"Risk Level: **{risk_label}**")

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 32}},
            title={"text": "Default Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if prediction else "#22c55e"},
                "steps": [
                    {"range": [0, 40], "color": "#dcfce7"},
                    {"range": [40, 60], "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Importance")

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        pre = pipeline.named_steps["pre"]
        cat_enc = pre.named_transformers_["cat"]
        num_names = artifact["numeric_features"]
        cat_names = cat_enc.get_feature_names_out(artifact["categorical_features"]).tolist()
        all_names = num_names + cat_names

        importances = clf.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": all_names, "importance": importances})
            .sort_values("importance", ascending=True)
            .tail(10)
        )

        fig2 = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker_color="#3b82f6",
        ))
        fig2.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Importance",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Fill in the applicant details in the sidebar and click **Analyze Risk**.")
