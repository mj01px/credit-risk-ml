<div align="center">

<br/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=30&pause=1000&color=FFFFFF&center=true&vCenter=true&width=600&lines=Credit+Risk+ML;Predict.+Evaluate.+Deploy." alt="Typing SVG" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
</p>

</div>

<br/>

---

## `~/about`

```python
credit_risk_ml = {
    "type":    "Machine Learning Pipeline · Web App",
    "stack":   ["Python 3.11", "scikit-learn", "pandas", "numpy", "Streamlit", "Plotly"],
    "models":  ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    "dataset": "Synthetic credit risk data — 2000 rows",
    "goal":    "Predict loan default probability from applicant data",
    "author":  "Mauro Junior · github.com/mj01px",
}
```

**Credit Risk ML** is a complete machine learning pipeline for predicting loan default probability. It covers the full data science workflow — from EDA and feature engineering to model comparison and an interactive Streamlit dashboard where users input applicant data and receive a default probability, risk classification, and feature importance breakdown.

```
credit-risk-ml/
├── data/
│   ├── generate.py          # Synthetic dataset generator
│   └── credit_risk.csv      # Generated dataset (2000 rows)
├── notebooks/
│   └── analysis.ipynb       # EDA + feature engineering + model comparison
├── models/
│   └── best_model.pkl       # Saved best model pipeline
├── app.py                   # Streamlit app
└── train.py                 # Training script
```

---

## `~/results`

<div align="center">

| Model | CV AUC | Notes |
|---|---|---|
| Logistic Regression | ≈ 0.80 | Baseline |
| Random Forest | ≈ 0.87 | Balanced performance |
| **Gradient Boosting** | **≈ 0.89** | **Best performer** |

</div>

**App outputs:** default probability gauge · risk classification · feature importance

---

## `~/getting-started`

```bash
git clone https://github.com/mj01px/credit-risk-ml.git
cd credit-risk-ml

# Setup
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt

# Generate data, train and launch
python data/generate.py
python train.py
streamlit run app.py
```

---

## `~/stack`

<div align="center">

| Layer | Technologies |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **ML** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **App** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| **Charts** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |
| **Notebook** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) |

</div>

---

<div align="center">
  <br/>
  <sub>
    Built by <a href="https://github.com/mj01px"><strong>Mauro Junior</strong></a> and <a href="https://github.com/JulioFranz)"><strong>Julio Franz</strong></a>
    &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/mauroapjunior/">LinkedIn</a>
  </sub>
  <br/><br/>
</div>
