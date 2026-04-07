"""Credit Risk Analyzer — Streamlit app."""
import math
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
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — full design system fiel ao HTML
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

/* tokens */
:root {
  --bg:           #f7f9fe;
  --surface:      #ffffff;
  --surface-lo:   #f1f4f8;
  --surface-mid:  #ebeef3;
  --surface-hi:   #e5e8ed;
  --border:       #e0e3e7;
  --outline:      #c0c7cd;

  --on-bg:        #181c1f;
  --on-sub:       #40484c;
  --on-muted:     #71787d;

  --primary:      #003345;
  --primary-cont: #004b63;
  --secondary:    #006a6a;
  --sec-cont:     #9deeed;
  --error:        #ba1a1a;
  --error-bg:     #ffdad6;
  --tert-cont:    #004e4a;
  --on-tert-cont: #62c2bb;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg) !important;
  color: var(--on-bg) !important;
  -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main .block-container {
  background: var(--bg) !important;
  padding-top: 0 !important;
  max-width: 100% !important;
}

.block-container { padding: 4px 40px 40px !important; }

/* Sidebar collapse */
[data-testid="stSidebar"]           { display: none !important; }
[data-testid="collapsedControl"]    { display: none !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }

/* Streamlit inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]   input {
  background: var(--surface-lo) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--on-bg) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  padding: 11px 14px !important;
  transition: border-color 150ms, box-shadow 150ms;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"]   input:focus {
  border-color: var(--secondary) !important;
  box-shadow: 0 0 0 3px rgba(0,106,106,.12) !important;
  outline: none !important;
}

[data-testid="stSelectbox"] > div > div {
  background: var(--surface-lo) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--on-bg) !important;
}

/* Field label */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"]   label,
[data-testid="stTextInput"]   label {
  font-size: 10px !important;
  font-weight: 700 !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  color: var(--on-muted) !important;
  margin-bottom: 4px !important;
}

/* Button */
.stButton > button {
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  padding: 14px 32px !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 8px !important;
  transition: box-shadow 150ms, transform 100ms !important;
  cursor: pointer !important;
}
.stButton > button:hover {
  box-shadow: 0 6px 20px rgba(0,51,69,.25) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: scale(.98) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar       { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--outline); border-radius: 99px; }

/* ── Custom classes ─────────────────────────────────── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}
.topbar-left { display: flex; align-items: center; gap: 32px; }
.topbar-brand {
  font-family: 'Manrope', sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: var(--primary) !important;
  letter-spacing: -.02em;
}
.topbar-nav { display: flex; gap: 24px; }
.topbar-nav a {
  font-size: 13px;
  font-weight: 500;
  color: var(--on-muted) !important;
  text-decoration: none;
  padding-bottom: 2px;
}
.topbar-nav a.active {
  color: var(--secondary) !important;
  border-bottom: 2px solid var(--secondary);
}

.eyebrow {
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--secondary) !important;
  margin: 0 0 6px;
}
.page-title {
  font-family: 'Manrope', sans-serif;
  font-size: 28px;
  font-weight: 900;
  color: var(--primary) !important;
  letter-spacing: -.03em;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.realtime-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--tert-cont);
  color: var(--on-tert-cont);
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 99px;
}
.page-header { margin-bottom: 32px; }

/* Form card */
.form-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 32px;
  margin-bottom: 16px;
}
.form-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 0;
  margin-bottom: 0;
}
.form-card-icon {
  width: 42px;
  height: 42px;
  background: rgba(0,75,99,.08);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.form-card-title {
  font-family: 'Manrope', sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: var(--primary) !important;
  margin: 0 0 3px;
}
.form-card-sub {
  font-size: 13px;
  color: var(--on-muted) !important;
  margin: 0;
}

/* Result card */
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 32px;
  position: relative;
  overflow: hidden;
  margin-bottom: 16px;
}
.result-card-title {
  font-family: 'Manrope', sans-serif;
  font-size: 16px;
  font-weight: 800;
  color: var(--primary) !important;
  margin: 0 0 28px;
}
.gauge-wrap { text-align: center; margin-bottom: 20px; }
.gauge-score {
  font-family: 'Manrope', sans-serif;
  font-size: 48px;
  font-weight: 900;
  color: var(--primary) !important;
  line-height: 1;
}
.gauge-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--on-muted) !important;
  margin-top: 4px;
}
.risk-badge-wrap { text-align: center; margin-bottom: 8px; }
.risk-badge {
  display: inline-block;
  padding: 8px 22px;
  border-radius: 99px;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.badge-low    { background: rgba(0,106,106,.08); color: var(--secondary); }
.badge-medium { background: #fff8e1;              color: #7a5500; }
.badge-high   { background: var(--error-bg);      color: var(--error); }
.confidence-line {
  text-align: center;
  font-size: 11px;
  font-weight: 500;
  color: var(--on-muted) !important;
  margin-bottom: 20px;
}

.rec-card {
  padding: 14px 16px;
  background: var(--surface-mid);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 10px;
}
.rec-card-eyebrow {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--on-muted) !important;
  margin: 0 0 6px;
}
.rec-value {
  font-family: 'Manrope', sans-serif;
  font-size: 18px;
  font-weight: 800;
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
}
.rec-value.approved { color: var(--secondary) !important; }
.rec-value.rejected { color: var(--error) !important; }

.insights-card {
  background: rgba(0,75,99,.06);
  border-radius: 10px;
  padding: 14px 16px;
  margin-top: 10px;
}
.insights-eyebrow {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--primary) !important;
  margin: 0 0 8px;
}
.insights-text {
  font-size: 12px;
  color: var(--on-sub) !important;
  font-style: italic;
  line-height: 1.6;
  margin: 0;
}

/* Glass panel */
.glass-card {
  background: rgba(255,255,255,.7);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,.3);
  border-radius: 14px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.glass-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 18px;
}
.glass-card-title {
  font-family: 'Manrope', sans-serif;
  font-size: 14px;
  font-weight: 700;
  color: var(--primary) !important;
  margin: 0;
}
.market-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 12px;
}
.market-stat-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--on-muted) !important;
  margin: 0 0 4px;
}
.market-stat-val {
  font-family: 'Manrope', sans-serif;
  font-size: 20px;
  font-weight: 800;
  color: var(--primary) !important;
  margin: 0;
}
.progress-track {
  width: 100%;
  height: 4px;
  background: var(--surface-hi);
  border-radius: 99px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--secondary);
  border-radius: 99px;
}

/* Feature importance section */
.section-header {
  font-family: 'Manrope', sans-serif;
  font-size: 16px;
  font-weight: 800;
  color: var(--primary) !important;
  margin: 32px 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* Placeholder state */
.placeholder-card {
  background: var(--surface);
  border: 1px dashed var(--outline);
  border-radius: 14px;
  padding: 48px 32px;
  text-align: center;
}
.placeholder-icon {
  width: 52px;
  height: 52px;
  background: var(--surface-lo);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
}
.placeholder-title {
  font-family: 'Manrope', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--primary) !important;
  margin: 0 0 8px;
}
.placeholder-desc {
  font-size: 13px;
  color: var(--on-muted) !important;
  max-width: 260px;
  margin: 0 auto;
  line-height: 1.6;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Load / train model
# ─────────────────────────────────────────────────────────────────────────────
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

if not MODEL_LOADED:
    st.error("Model failed to load.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  Top bar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <span class="topbar-brand">Credit Risk Analyzer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <p class="eyebrow">Asset Assessment</p>
  <h1 class="page-title">
    Credit Risk Predictor
  </h1>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Two-column layout
# ─────────────────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([7, 5], gap="large")

# ── LEFT: form card ───────────────────────────────────────────────────────────
with col_form:
    st.markdown("""
    <div class="form-card">
      <div class="form-card-header">
        <div class="form-card-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#003345" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </div>
        <div>
          <p class="form-card-title">Applicant Data Entry</p>
          <p class="form-card-sub">Provide details for comprehensive risk assessment</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        r1a, r1b = st.columns(2)
        with r1a:
            age = st.number_input("Age", min_value=18, max_value=75, value=32)
        with r1b:
            annual_income = st.number_input("Annual Income", min_value=12_000, max_value=200_000, value=85_000, step=1_000, format="%d")
            st.caption(f"$ {annual_income:,.2f}")

        r2a, r2b = st.columns(2)
        with r2a:
            employment_years = st.number_input("Employment Length (Years)", min_value=0, max_value=40, value=7)
        with r2b:
            loan_amount = st.number_input("Loan Amount", min_value=1_000, max_value=60_000, value=25_000, step=500, format="%d")
            st.caption(f"$ {loan_amount:,.2f}")

        r3a, r3b = st.columns(2)
        with r3a:
            loan_purpose = st.selectbox(
                "Loan Intent",
                ["personal", "car", "education", "home_improvement", "medical"],
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with r3b:
            loan_duration_months = st.selectbox(
                "Duration",
                [12, 24, 36, 48, 60],
                index=2,
                format_func=lambda x: f"{x} months",
            )

        r4a, r4b = st.columns(2)
        with r4a:
            credit_history = st.selectbox(
                "Credit History",
                ["good", "fair", "poor"],
                format_func=str.title,
            )
        with r4b:
            num_credit_lines = st.number_input("Credit Lines", min_value=0, max_value=15, value=3)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Risk Analysis", use_container_width=False)

# ── RIGHT: result card ────────────────────────────────────────────────────────
with col_result:
    if not predict_btn:
        st.markdown("""
        <div class="placeholder-card">
          <div class="placeholder-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#71787d" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 8v4l3 3"/>
            </svg>
          </div>
          <p class="placeholder-title">Awaiting Analysis</p>
          <p class="placeholder-desc" style="text-align:center;margin:0 auto">
            Complete the applicant form and click
            <strong style="color:var(--primary);font-weight:700">Run Risk Analysis</strong>
            to generate a prediction.
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Compute prediction ────────────────────────────────────────────────
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
        confidence = abs(proba - 0.5) * 2  # 0-1 distance from decision boundary
        conf_pct   = round(50 + confidence * 50, 1)

        # risk tier
        if proba < 0.35:
            risk_label, badge_cls = "Low Risk Tier",    "badge-low"
        elif proba < 0.60:
            risk_label, badge_cls = "Medium Risk Tier", "badge-medium"
        else:
            risk_label, badge_cls = "High Risk Tier",   "badge-high"

        approved = not prediction
        rec_cls   = "approved" if approved else "rejected"
        rec_text  = "Approved"  if approved else "Rejected"
        rec_icon  = (
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#006a6a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
            if approved else
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ba1a1a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
        )

        # Circular SVG gauge params
        r          = 72
        circ       = 2 * math.pi * r
        fill_pct   = proba
        dash_offset= circ * (1 - fill_pct)
        stroke_col = "#ba1a1a" if prediction else "#006a6a"

        insights_text = (
            f'Applicant presents elevated debt-to-income ratio of {dti:.1%}. '
            f'Employment tenure and credit history are significant risk contributors. '
            f'Manual review is recommended before final decision.'
        ) if prediction else (
            f'Applicant demonstrates strong financial indicators. '
            f'Debt-to-income ratio of {dti:.1%} falls within acceptable bounds. '
            f'Employment tenure and credit history support approval.'
        )

        st.markdown(f"""
        <div class="result-card">
          <div style="position:absolute;top:0;right:0;padding:20px;opacity:.07">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="#006a6a"><path d="M12 1l3.09 6.26L22 8.27l-5 4.87 1.18 6.88L12 16.9l-6.18 3.12L7 13.14 2 8.27l6.91-1.01L12 1z"/></svg>
          </div>
          <p class="result-card-title">Prediction Result</p>

          <!-- Circular gauge -->
          <div class="gauge-wrap">
            <div style="position:relative;width:160px;height:160px;margin:0 auto 20px;">
              <svg width="160" height="160" viewBox="0 0 160 160" style="transform:rotate(-90deg);display:block;">
                <circle cx="80" cy="80" r="{r}" fill="transparent"
                  stroke="#e0e3e7" stroke-width="12"/>
                <circle cx="80" cy="80" r="{r}" fill="transparent"
                  stroke="{stroke_col}" stroke-width="12"
                  stroke-dasharray="{circ:.2f}"
                  stroke-dashoffset="{dash_offset:.2f}"
                  stroke-linecap="round"/>
              </svg>
              <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">
                <p class="gauge-score">{proba:.0%}</p>
                <p class="gauge-label">Default Probability</p>
              </div>
            </div>
          </div>

          <!-- Badge + confidence -->
          <div class="risk-badge-wrap">
            <span class="risk-badge {badge_cls}">{risk_label}</span>
          </div>
          <p class="confidence-line">Model Confidence: {conf_pct:.1f}%</p>

          <!-- Recommendation -->
          <div class="rec-card">
            <p class="rec-card-eyebrow">Recommendation</p>
            <p class="rec-value {rec_cls}">{rec_icon}{rec_text}</p>
          </div>

          <!-- ML Insights -->
          <div class="insights-card">
            <p class="insights-eyebrow">ML Insights</p>
            <p class="insights-text">"{insights_text}"</p>
          </div>
        </div>

        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Feature importance (full width, only after analysis)
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        st.markdown("""
        <p class="section-header">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#003345" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6"  y1="20" x2="6"  y2="14"/>
          </svg>
          Feature Importance
        </p>
        """, unsafe_allow_html=True)

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
        labels = [f.replace("_", " ").title() for f in fi_df["feature"]]

        fig = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=labels,
            orientation="h",
            marker_color="#006a6a",
            marker_line_width=0,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=20),
            xaxis=dict(
                title="Importance Score",
                color="#71787d",
                gridcolor="#e0e3e7",
                showline=False,
                tickfont={"size": 11, "family": "Inter", "color": "#71787d"},
                title_font={"size": 11, "color": "#71787d"},
            ),
            yaxis=dict(
                color="#40484c",
                gridcolor="#e0e3e7",
                tickfont={"size": 11, "family": "Inter", "color": "#40484c"},
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True)
