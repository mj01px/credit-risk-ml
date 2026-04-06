"""Credit Risk Analyzer — Streamlit app."""
import joblib
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "models/best_model.pkl"

st.set_page_config(
    page_title="Credit Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

  /* ── Reset & tokens ─────────────────────────────────────────── */
  :root {
    --bg:           #f7f9fe;
    --surface:      #ffffff;
    --surface-lo:   #f1f4f8;
    --surface-mid:  #ebeef3;
    --surface-hi:   #e5e8ed;
    --border:       #c0c7cd;
    --border-sub:   #e0e3e7;

    --on-bg:        #181c1f;
    --on-sub:       #40484c;
    --on-muted:     #71787d;

    --primary:      #003345;
    --primary-cont: #004b63;
    --secondary:    #006a6a;
    --error:        #ba1a1a;
    --error-bg:     #ffdad6;
    --success:      #006a6a;
    --success-bg:   #9deeed30;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--on-bg) !important;
  }

  /* App shell */
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] { background: var(--bg) !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border-sub) !important;
  }
  [data-testid="stSidebar"] section { padding-top: 0 !important; }
  [data-testid="stSidebar"] label   { color: var(--on-sub)   !important; font-size: 12px !important; font-weight: 500 !important; }
  [data-testid="stSidebar"] p       { color: var(--on-muted) !important; }

  /* Inputs */
  input, select, textarea {
    background: var(--surface-lo) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 6px !important;
    color: var(--on-bg) !important;
    font-size: 13px !important;
  }
  input:focus, select:focus { border-color: var(--secondary) !important; box-shadow: 0 0 0 3px #006a6a18 !important; }

  /* Slider */
  [data-testid="stSlider"] [role="slider"]          { background: var(--secondary) !important; border-color: var(--secondary) !important; }
  [data-testid="stSlider"] > div > div > div        { background: var(--border-sub) !important; }
  [data-testid="stSlider"] > div > div > div > div  { background: var(--secondary) !important; }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: var(--surface-lo) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 6px !important;
  }

  /* Button */
  .stButton > button {
    background: var(--primary) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: .01em !important;
    padding: 11px 0 !important;
    transition: background 150ms ease !important;
  }
  .stButton > button:hover { background: #004b63 !important; }
  .stButton > button:active { transform: scale(.98) !important; }

  /* Expander */
  [data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
  }
  [data-testid="stExpander"] summary       { padding: 12px 16px !important; }
  [data-testid="stExpander"] summary span  { color: var(--on-sub) !important; font-size: 13px !important; font-weight: 500 !important; }

  /* Metric */
  [data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sub) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.04) !important;
  }
  [data-testid="stMetricValue"] { color: var(--primary) !important; font-family: 'Manrope', sans-serif !important; font-size: 28px !important; font-weight: 800 !important; }
  [data-testid="stMetricLabel"] { color: var(--on-muted) !important; font-size: 10px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .08em; }

  /* Divider */
  hr { border-color: var(--border-sub) !important; margin: 20px 0 !important; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

  /* ── Custom components ───────────────────────────────────────── */
  .page-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--secondary) !important;
    margin: 0 0 6px;
  }
  .page-title {
    font-family: 'Manrope', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--primary) !important;
    margin: 0 0 2px;
    letter-spacing: -.02em;
    line-height: 1.2;
  }
  .page-sub {
    font-size: 13px;
    color: var(--on-muted) !important;
    margin: 0 0 28px;
  }
  .ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #9deeed30;
    color: var(--secondary);
    border: 1px solid #9deeed;
    border-radius: 99px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 3px 10px;
    margin-left: 10px;
    vertical-align: middle;
  }
  .section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--on-muted) !important;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-sub);
  }
  .sidebar-brand {
    padding: 24px 4px 20px;
    border-bottom: 1px solid var(--border-sub);
    margin-bottom: 4px;
  }
  .brand-title {
    font-family: 'Manrope', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: var(--primary) !important;
    letter-spacing: -.02em;
    margin: 0 0 2px;
  }
  .brand-sub {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--on-muted) !important;
    margin: 0;
  }

  /* Result card */
  .result-card {
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
  }
  .result-card.danger {
    background: #fff5f5;
    border: 1px solid #f5c6c6;
  }
  .result-card.success {
    background: #f0faf8;
    border: 1px solid #a5d6d6;
  }
  .result-eyebrow {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin: 0 0 6px;
  }
  .result-card.danger  .result-eyebrow { color: #ba1a1a; }
  .result-card.success .result-eyebrow { color: #006a6a; }
  .result-verdict {
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 800;
    margin: 0 0 6px;
    letter-spacing: -.02em;
  }
  .result-card.danger  .result-verdict { color: var(--primary); }
  .result-card.success .result-verdict { color: var(--primary); }
  .result-desc {
    font-size: 13px;
    color: var(--on-sub) !important;
    margin: 0 0 16px;
    line-height: 1.5;
  }
  .risk-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 99px;
  }
  .badge-vlow  { background:#9deeed30; color:#006a6a; border:1px solid #84d4d3; }
  .badge-low   { background:#c8f5e8;   color:#005740; border:1px solid #84d4d3; }
  .badge-med   { background:#fff8e1;   color:#7a5500; border:1px solid #f0c060; }
  .badge-high  { background:#ffdad6;   color:#ba1a1a; border:1px solid #f5c6c6; }
  .badge-vhigh { background:#ffdad6;   color:#8b0000; border:1px solid #e08080; }

  /* Stat grid */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 0;
  }
  .stat-item {
    background: var(--surface);
    border: 1px solid var(--border-sub);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }
  .stat-val {
    font-family: 'Manrope', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--primary) !important;
    letter-spacing: -.02em;
  }
  .stat-lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--on-muted) !important;
    margin-top: 4px;
  }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 80px 24px;
    background: var(--surface);
    border: 1px solid var(--border-sub);
    border-radius: 12px;
  }
  .empty-icon {
    width: 52px;
    height: 52px;
    background: var(--surface-lo);
    border: 1px solid var(--border-sub);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 18px;
  }
  .empty-title {
    font-family: 'Manrope', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: var(--primary) !important;
    margin: 0 0 8px;
  }
  .empty-desc {
    font-size: 13px;
    color: var(--on-muted) !important;
    max-width: 340px;
    margin: 0 auto;
    line-height: 1.65;
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
    <div class="sidebar-brand">
      <p class="brand-title">Credit Risk Analyzer</p>
      <p class="brand-sub">Institutional Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Applicant</p>', unsafe_allow_html=True)
    age              = st.number_input("Age", 18, 75, 35)
    annual_income    = st.number_input("Annual Income ($)", 12_000, 200_000, 55_000, 1_000, format="%d")
    employment_years = st.number_input("Years Employed", 0, 40, 5)

    st.markdown('<p class="section-label">Loan</p>', unsafe_allow_html=True)
    loan_amount          = st.number_input("Loan Amount ($)", 1_000, 60_000, 15_000, 500, format="%d")
    loan_duration_months = st.selectbox("Duration", [12, 24, 36, 48, 60], index=2,
                                         format_func=lambda x: f"{x} months")
    loan_purpose         = st.selectbox("Purpose",
                                         ["personal", "car", "education", "home_improvement", "medical"],
                                         format_func=lambda x: x.replace("_", " ").title())

    st.markdown('<p class="section-label">Credit Profile</p>', unsafe_allow_html=True)
    credit_history   = st.selectbox("Credit History", ["good", "fair", "poor"], format_func=str.title)
    num_credit_lines = st.number_input("Credit Lines", 0, 15, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyze Risk", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
if not MODEL_LOADED:
    st.error("Model failed to load. Check the logs.")
    st.stop()

auc_val = metrics.get("test_auc", "—")

st.markdown(f"""
<p class="page-eyebrow">Credit Intelligence Platform</p>
<h1 class="page-title">
  Default Risk Assessment
  <span class="ai-badge">ML · AUC {auc_val}</span>
</h1>
<p class="page-sub">Fill in the applicant profile and loan details to generate a risk prediction.</p>
""", unsafe_allow_html=True)

with st.expander("Model details", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", model_name)
    c2.metric("Test AUC",  metrics.get("test_auc", "—"))
    c3.metric("CV AUC",    metrics.get("cv_auc",   "—"))
    c4.metric("Accuracy",  f"{metrics.get('accuracy', 0):.1%}")

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

    card_cls      = "danger"  if prediction else "success"
    verdict_eye   = "High Risk — Likely to Default"    if prediction else "Low Risk — Unlikely to Default"
    verdict_title = "This applicant presents elevated default risk." if prediction \
                    else "This applicant presents low default risk."
    verdict_desc  = "The model identifies significant risk factors in this profile. Detailed review is recommended before approval." \
                    if prediction else \
                    "No significant risk factors detected. The applicant's profile is consistent with low default probability."

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f"""
        <div class="result-card {card_cls}">
          <p class="result-eyebrow">{verdict_eye}</p>
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
        needle   = "#ba1a1a" if prediction else "#006a6a"
        step_hi  = "#ffdad6" if prediction else "#9deeed40"
        step_mid = "#fff8e1"
        step_lo  = "#9deeed40" if not prediction else "#ffdad620"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 38, "color": "#003345", "family": "Manrope"}},
            title={"text": "Default Probability", "font": {"color": "#71787d", "size": 12, "family": "Inter"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#c0c7cd",
                    "tickfont": {"color": "#71787d", "size": 10, "family": "Inter"},
                },
                "bar":      {"color": needle, "thickness": 0.22},
                "bgcolor":  "#ffffff",
                "bordercolor": "#e0e3e7",
                "steps": [
                    {"range": [0,  40], "color": "#f0faf8"},
                    {"range": [40, 60], "color": "#fffbf0"},
                    {"range": [60, 100],"color": "#fff5f5"},
                ],
                "threshold": {
                    "line": {"color": "#40484c", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(
            height=280,
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
            marker_color="#006a6a",
            marker_line_width=0,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            height=320,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(
                title="Importance",
                color="#71787d",
                gridcolor="#e0e3e7",
                showline=False,
                tickfont={"size": 11, "family": "Inter"},
                title_font={"size": 11, "color": "#71787d"},
            ),
            yaxis=dict(
                color="#40484c",
                gridcolor="#e0e3e7",
                tickfont={"size": 11, "family": "Inter"},
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "#40484c"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.caption("Feature importance not available for this model type.")

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#71787d" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <rect x="2" y="7" width="20" height="14" rx="2"/>
          <path d="M16 7V5a2 2 0 0 0-4 0v2"/>
          <path d="M8 7V5a2 2 0 0 1 4 0"/>
          <line x1="12" y1="12" x2="12" y2="16"/>
          <line x1="10" y1="14" x2="14" y2="14"/>
        </svg>
      </div>
      <p class="empty-title">No Analysis Yet</p>
      <p class="empty-desc">
        Fill in the applicant profile and loan details in the sidebar,
        then click <strong style="color:#003345;font-weight:600">Analyze Risk</strong>
        to generate a default probability report.
      </p>
    </div>
    """, unsafe_allow_html=True)
