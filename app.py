"""Streamlit app for credit risk prediction."""
import joblib
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "models/best_model.pkl"

st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f1117; }
  [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }

  .card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .risk-high {
    background: linear-gradient(135deg, #2d1515 0%, #1a0a0a 100%);
    border: 1px solid #f85149;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .risk-low {
    background: linear-gradient(135deg, #0d2318 0%, #061a10 100%);
    border: 1px solid #3fb950;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .risk-title { font-size: 22px; font-weight: 700; margin: 0; }
  .risk-high .risk-title { color: #f85149; }
  .risk-low  .risk-title { color: #3fb950; }
  .risk-sub  { font-size: 13px; color: #8b949e; margin-top: 4px; }

  .stat-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }
  .stat-box {
    flex: 1;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }
  .stat-val { font-size: 24px; font-weight: 700; color: #e6edf3; }
  .stat-lbl { font-size: 11px; color: #6e7681; text-transform: uppercase; letter-spacing: .05em; margin-top: 4px; }

  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 12px;
  }
  .badge-vlow  { background:#0d2318; color:#3fb950; border:1px solid #3fb950; }
  .badge-low   { background:#0f2d1a; color:#56d364; border:1px solid #56d364; }
  .badge-med   { background:#2d2208; color:#e3b341; border:1px solid #e3b341; }
  .badge-high  { background:#2d1515; color:#f85149; border:1px solid #f85149; }
  .badge-vhigh { background:#1a0505; color:#ff7b72; border:1px solid #ff7b72; }

  .section-title {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 12px;
  }
  h1, h2, h3 { color: #e6edf3 !important; }
  p, label, span { color: #8b949e; }
  .stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px !important;
  }
  .stButton > button:hover { background: #2ea043 !important; }
  div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
  }
  div[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model, please wait...")
def load_model():
    try:
        artifact = joblib.load(MODEL_PATH)
        artifact["pipeline"].predict_proba(pd.DataFrame([{
            "age": 30, "annual_income": 50000, "employment_years": 5,
            "loan_amount": 10000, "loan_duration_months": 36,
            "num_credit_lines": 3, "debt_to_income_ratio": 0.2,
            "loan_purpose": "personal", "credit_history": "good",
        }]))
        return artifact
    except Exception:
        pass

    data_path = "data/credit_risk.csv"
    if not os.path.exists(data_path):
        sys.path.insert(0, ".")
        from data.generate import generate
        os.makedirs("data", exist_ok=True)
        generate().to_csv(data_path, index=False)

    os.makedirs("models", exist_ok=True)
    import subprocess
    subprocess.run([sys.executable, "train.py"], check=True)
    return joblib.load(MODEL_PATH)


try:
    artifact  = load_model()
    pipeline  = artifact["pipeline"]
    model_name = artifact["model_name"]
    metrics   = artifact["metrics"]
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk Analyzer")
    st.markdown("<p style='font-size:13px;color:#6e7681;margin-top:-8px'>Fill in the fields and click Analyze</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p class='section-title'>👤 Applicant</p>", unsafe_allow_html=True)
    age              = st.slider("Age", 18, 75, 35)
    annual_income    = st.number_input("Annual Income ($)", 12_000, 200_000, 55_000, 1_000, format="%d")
    employment_years = st.slider("Years Employed", 0, 40, 5)

    st.markdown("---")
    st.markdown("<p class='section-title'>🏦 Loan</p>", unsafe_allow_html=True)
    loan_amount          = st.number_input("Loan Amount ($)", 1_000, 60_000, 15_000, 500, format="%d")
    loan_duration_months = st.selectbox("Duration", [12, 24, 36, 48, 60],
                                         index=2, format_func=lambda x: f"{x} months")
    loan_purpose         = st.selectbox("Purpose",
                                         ["personal", "car", "education", "home_improvement", "medical"],
                                         format_func=lambda x: x.replace("_", " ").title())

    st.markdown("---")
    st.markdown("<p class='section-title'>📊 Credit Profile</p>", unsafe_allow_html=True)
    credit_history   = st.selectbox("Credit History", ["good", "fair", "poor"], format_func=str.title)
    num_credit_lines = st.slider("Credit Lines", 0, 15, 3)

    st.markdown("---")
    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Credit Risk Analyzer")
st.markdown("<p style='color:#8b949e;margin-top:-12px;margin-bottom:24px'>Predict loan default probability using machine learning</p>", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error("Model failed to load. Check the logs.")
    st.stop()

# Model info strip
with st.expander("ℹ️ Model Info", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm",  model_name)
    c2.metric("Test AUC",   metrics.get("test_auc", "—"))
    c3.metric("CV AUC",     metrics.get("cv_auc", "—"))
    c4.metric("Accuracy",   f"{metrics.get('accuracy', 0):.1%}")

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    dti = round((loan_amount / loan_duration_months * 12) / annual_income, 3)

    input_df = pd.DataFrame([{
        "age": age, "annual_income": annual_income,
        "employment_years": employment_years, "loan_amount": loan_amount,
        "loan_duration_months": loan_duration_months,
        "num_credit_lines": num_credit_lines,
        "debt_to_income_ratio": dti,
        "loan_purpose": loan_purpose, "credit_history": credit_history,
    }])

    proba      = pipeline.predict_proba(input_df)[0, 1]
    prediction = int(proba >= 0.5)

    risk_label, badge_cls = (
        ("Very Low",  "badge-vlow")  if proba < 0.25 else
        ("Low",       "badge-low")   if proba < 0.40 else
        ("Medium",    "badge-med")   if proba < 0.55 else
        ("High",      "badge-high")  if proba < 0.75 else
        ("Very High", "badge-vhigh")
    )

    verdict_cls   = "risk-high" if prediction else "risk-low"
    verdict_title = "HIGH RISK — Likely to Default" if prediction else "LOW RISK — Unlikely to Default"
    verdict_sub   = "This applicant has a high probability of defaulting." if prediction else "This applicant is unlikely to default on the loan."

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f"""
        <div class="{verdict_cls}">
          <p class="risk-title">{verdict_title}</p>
          <p class="risk-sub">{verdict_sub}</p>
          <span class="badge {badge_cls}">Risk Level: {risk_label}</span>
        </div>
        <div class="stat-row">
          <div class="stat-box">
            <div class="stat-val">{proba:.1%}</div>
            <div class="stat-lbl">Default Probability</div>
          </div>
          <div class="stat-box">
            <div class="stat-val">{dti:.1%}</div>
            <div class="stat-lbl">Debt-to-Income</div>
          </div>
          <div class="stat-box">
            <div class="stat-val">${loan_amount:,}</div>
            <div class="stat-lbl">Loan Amount</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        bar_color = "#f85149" if prediction else "#3fb950"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
            title={"text": "Default Probability", "font": {"color": "#8b949e", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6e7681",
                         "tickfont": {"color": "#6e7681"}},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#0d1117",
                "bordercolor": "#21262d",
                "steps": [
                    {"range": [0,  40], "color": "#0d2318"},
                    {"range": [40, 60], "color": "#1c1a0a"},
                    {"range": [60, 100],"color": "#2d1515"},
                ],
                "threshold": {"line": {"color": "#e6edf3", "width": 2},
                              "thickness": 0.8, "value": 50},
            },
        ))
        fig.update_layout(
            height=280,
            margin=dict(t=40, b=0, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf3"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-title'>Feature Importance</p>", unsafe_allow_html=True)

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        pre       = pipeline.named_steps["pre"]
        cat_enc   = pre.named_transformers_["cat"]
        num_names = artifact["numeric_features"]
        cat_names = cat_enc.get_feature_names_out(artifact["categorical_features"]).tolist()
        all_names = num_names + cat_names

        fi_df = (
            pd.DataFrame({"feature": all_names, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=True)
            .tail(10)
        )
        colors = ["#388bfd" if f not in input_df.columns else "#f78166" for f in fi_df["feature"]]

        fig2 = go.Figure(go.Bar(
            x=fi_df["importance"], y=fi_df["feature"],
            orientation="h", marker_color="#388bfd",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            height=340,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(title="Importance", color="#6e7681", gridcolor="#21262d", showline=False),
            yaxis=dict(color="#8b949e", gridcolor="#21262d"),
            plot_bgcolor="#161b22",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#8b949e"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Feature importance not available for this model type.")

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:56px;margin-bottom:16px">💳</div>
      <h3 style="color:#e6edf3;margin-bottom:8px">Ready to Analyze</h3>
      <p style="color:#6e7681;max-width:400px;margin:0 auto">
        Fill in the applicant's profile and loan details in the sidebar,
        then click <strong style="color:#e6edf3">Analyze Risk</strong> to get the prediction.
      </p>
    </div>
    """, unsafe_allow_html=True)
