# Credit Risk ML

Machine learning pipeline to predict loan default probability. Includes full EDA, feature engineering, model comparison, and an interactive Streamlit app.

## Demo

The app takes applicant data (income, loan amount, credit history, etc.) and returns:
- Default probability with a gauge chart
- Risk level classification
- Feature importance breakdown

## Stack

- **Python 3.11**
- **pandas + numpy** — data processing
- **scikit-learn** — model training and evaluation
- **Streamlit** — interactive web app
- **Plotly** — charts and gauge

## Models Compared

| Model | CV AUC | Notes |
|-------|--------|-------|
| Logistic Regression | ~0.80 | Baseline |
| Random Forest | ~0.87 | Good balance |
| Gradient Boosting | ~0.89 | Best overall |

## Quick Start

```bash
git clone https://github.com/mj01px/credit-risk-ml
cd credit-risk-ml
pip install -r requirements.txt
```

**1. Generate dataset and train models:**
```bash
python data/generate.py
python train.py
```

**2. Run the app:**
```bash
streamlit run app.py
```

## Project Structure

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
├── train.py                 # Training script
└── requirements.txt
```

## Features

- **EDA:** target distribution, feature distributions, correlation matrix
- **Feature Engineering:** debt-to-income ratio, age groups, income brackets
- **Model Comparison:** cross-validated AUC, ROC curves, metrics bar chart
- **App:** real-time prediction with gauge, risk level, feature importance

## Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo and set `app.py` as the entry point
4. Add a startup command to generate data and train: `python data/generate.py && python train.py`
