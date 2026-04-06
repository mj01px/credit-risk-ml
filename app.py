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
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system (mirrors game-rent dark palette) ────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg:         #141414;
    --surface:    #1C1C1C;
    --surface-2:  #242424;
    --border:     #3C4043;
    --border-sub: #2A2A2A;

    --text:       #E8EAED;
    --text-sub:   #9AA0A6;
    --text-muted: #5F6368;

    --accent:     #8AB4F8;
    --danger:     #F28B82;
    --success:    #81C995;
    --warning:    #FDD663;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* App background */
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] { background: var(--bg) !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border-sub) !important;
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] div { color: var(--text-sub) !important; }

  /* Inputs */
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] select,
  [data-testid="stSidebar"] textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
  }

  /* Slider */
  [data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
  [data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
  [data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

  /* Button */
  .stButton > button {
    background: var(--accent) !important;
    color: #0d1b2a !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .01em !important;
    padding: 10px 0 !important;
    transition: opacity 150ms ease !important;
  }
  .stButton > button:hover { opacity: .85 !important; }

  /* Expander */
  [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] summary span { color: var(--text-sub) !important; }

  /* Metric */
  [data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
  }
  [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 22px !important; }
  [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 11px !important; }

  /* Divider */
  hr { border-color: var(--border-sub) !important; }

  /* Global text */
  h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; }
  p, span, label  { color: var(--text-sub) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

  /* ── Custom components ── */
  .page-title {
    font-size: 24px;
    font-weight: 700;
    color: var(--text) !important;
    margin: 0 0 4px;
    letter-spacing: -.02em;
  }
  .page-sub {
    font-size: 13px;
    color: var(--text-muted) !important;
    margin: 0 0 24px;
  }
  .section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin: 20px 0 8px;
  }
  .result-card {
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 16px;
  }
  .result-card.danger {
    background: #1f0f0f;
    border: 1px solid #5c2323;
  }
  .result-card.success {
    background: #0b1a10;
    border: 1px solid #1e4d2b;
  }
  .result-verdict {
    font-size: 17px;
    font-weight: 700;
    margin: 0 0 4px;
  }
  .result-card.danger  .result-verdict { color: var(--danger);  }
  .result-card.success .result-verdict { color: var(--success); }
  .result-desc {
    font-size: 13px;
    color: var(--text-muted) !important;
    margin: 0 0 16px;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .stat-item {
    background: var(--surface-2);
    border: 1px solid var(--border-sub);
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
  }
  .stat-val {
    font-size: 20px;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -.01em;
  }
  .stat-lbl {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-top: 3px;
  }
  .risk-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
  }
  .badge-vlow  { background:#0b1a10; color:var(--success); border:1px solid #1e4d2b; }
  .badge-low   { background:#0d2015; color:#57D98A; border:1px solid #1a4a27; }
  .badge-med   { background:#1c1a0a; color:var(--warning); border:1px solid #5c4a10; }
  .badge-high  { background:#1f0f0f; color:var(--danger);  border:1px solid #5c2323; }
  .badge-vhigh { background:#170505; color:#FF9F9A;         border:1px solid #6b1a1a; }
  .empty-state {
    text-align: center;
    padding: 80px 24px;
  }
  .empty-icon {
    width: 48px;
    height: 48px;
    background: var(--surface);
    border: 1px solid var(--border-sub);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 20px;
  }
  .empty-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text) !important;
    margin: 0 0 6px;
  }
  .empty-desc {
    font-size: 13px;
    color: var(--text-muted) !important;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.6;
  }
  .sidebar-logo {
    font-size: 15px;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -.01em;
    margin-bottom: 2px;
  }
  .sidebar-tagline {
    font-size: 11px;
    color: var(--text-muted) !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Preparing model...")
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
    artifact   = load_model()
    pipeline   = artifact["pipeline"]
    model_name = artifact["model_name"]
    metrics    = artifact["metrics"]
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 20px">
      <div class="sidebar-logo">Credit Risk Analyzer</div>
      <div class="sidebar-tagline">Default probability predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Applicant</div>', unsafe_allow_html=True)
    age              = st.slider("Age", 18, 75, 35, label_visibility="collapsed")
    st.caption(f"Age: {age} years")
    annual_income    = st.number_input("Annual Income ($)", 12_000, 200_000, 55_000, 1_000, format="%d")
    employment_years = st.slider("Years Employed", 0, 40, 5)

    st.markdown('<div class="section-label">Loan</div>', unsafe_allow_html=True)
    loan_amount          = st.number_input("Loan Amount ($)", 1_000, 60_000, 15_000, 500, format="%d")
    loan_duration_months = st.selectbox("Duration", [12, 24, 36, 48, 60], index=2,
                                         format_func=lambda x: f"{x} months")
    loan_purpose         = st.selectbox("Purpose",
                                         ["personal", "car", "education", "home_improvement", "medical"],
                                         format_func=lambda x: x.replace("_", " ").title())

    st.markdown('<div class="section-label">Credit Profile</div>', unsafe_allow_html=True)
    credit_history   = st.selectbox("Credit History", ["good", "fair", "poor"], format_func=str.title)
    num_credit_lines = st.slider("Credit Lines", 0, 15, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyze Risk", use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">Credit Risk Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Predict loan default probability using machine learning</p>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error("Model failed to load. Check the logs.")
    st.stop()

with st.expander("Model details", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", model_name)
    c2.metric("Test AUC",  metrics.get("test_auc", "—"))
    c3.metric("CV AUC",    metrics.get("cv_auc",   "—"))
    c4.metric("Accuracy",  f"{metrics.get('accuracy', 0):.1%}")

st.markdown("---")

# ── Prediction result ─────────────────────────────────────────────────────────
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

    card_cls      = "danger"  if prediction else "success"
    verdict_title = "High Risk — Likely to Default"    if prediction else "Low Risk — Unlikely to Default"
    verdict_desc  = "This applicant has a high probability of defaulting on the loan." if prediction \
                    else "This applicant is unlikely to default based on the provided data."

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f"""
        <div class="result-card {card_cls}">
          <p class="result-verdict">{verdict_title}</p>
          <p class="result-desc">{verdict_desc}</p>
          <span class="risk-badge {badge_cls}">{risk_label} Risk</span>
        </div>
        <div class="stat-grid">
          <div class="stat-item">
            <div class="stat-val">{proba:.1%}</div>
            <div class="stat-lbl">Default Probability</div>
          </div>
          <div class="stat-item">
            <div class="stat-val">{dti:.1%}</div>
            <div class="stat-lbl">Debt / Income</div>
          </div>
          <div class="stat-item">
            <div class="stat-val">${loan_amount:,}</div>
            <div class="stat-lbl">Loan Amount</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        needle_color = "#F28B82" if prediction else "#81C995"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 34, "color": "#E8EAED", "family": "Inter"}},
            title={"text": "Default Probability", "font": {"color": "#5F6368", "size": 12, "family": "Inter"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#3C4043",
                    "tickfont": {"color": "#5F6368", "size": 10},
                },
                "bar": {"color": needle_color, "thickness": 0.22},
                "bgcolor": "#141414",
                "bordercolor": "#2A2A2A",
                "steps": [
                    {"range": [0,  40], "color": "#0b1a10"},
                    {"range": [40, 60], "color": "#1c1a0a"},
                    {"range": [60, 100],"color": "#1f0f0f"},
                ],
                "threshold": {
                    "line": {"color": "#9AA0A6", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(
            height=270,
            margin=dict(t=50, b=10, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">Feature Importance</p>', unsafe_allow_html=True)

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

        fig2 = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker_color="#8AB4F8",
            marker_line_width=0,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(
                title="Importance", color="#5F6368",
                gridcolor="#2A2A2A", showline=False, tickfont={"size": 11},
            ),
            yaxis=dict(color="#9AA0A6", gridcolor="#2A2A2A", tickfont={"size": 11}),
            plot_bgcolor="#1C1C1C",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "#9AA0A6"},
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Feature importance not available for this model type.")

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5F6368" stroke-width="2">
          <rect x="2" y="5" width="20" height="14" rx="2"/>
          <path d="M2 10h20"/>
        </svg>
      </div>
      <p class="empty-title">Ready to analyze</p>
      <p class="empty-desc">
        Fill in the applicant profile and loan details in the sidebar,
        then click <strong style="color:#E8EAED">Analyze Risk</strong> to get the prediction.
      </p>
    </div>
    """, unsafe_allow_html=True)
