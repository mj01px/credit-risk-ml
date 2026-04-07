"""Train and compare credit risk models, save the best one."""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/credit_risk.csv"
MODEL_PATH = "models/best_model.pkl"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_duration_months",
    "num_credit_lines",
    "debt_to_income_ratio",
]
CATEGORICAL_FEATURES = ["loan_purpose", "credit_history"]
TARGET = "default"

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

def build_pipelines() -> dict[str, Pipeline]:
    pre = build_preprocessor()
    return {
        "Logistic Regression": Pipeline(
            [("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "Random Forest": Pipeline(
            [("pre", build_preprocessor()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]
        ),
        "Gradient Boosting": Pipeline(
            [("pre", build_preprocessor()), ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))]
        ),
    }

def evaluate(pipelines: dict, X_train, X_test, y_train, y_test) -> dict:
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, pipe in pipelines.items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "cv_auc": round(cv_auc, 4),
            "test_auc": round(roc_auc_score(y_test, y_proba), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
        }
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"  CV AUC:   {results[name]['cv_auc']}")
        print(f"  Test AUC: {results[name]['test_auc']}")
        print(f"  Accuracy: {results[name]['accuracy']}")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return results

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Default rate: {y.mean():.1%}")

    pipelines = build_pipelines()
    results = evaluate(pipelines, X_train, X_test, y_train, y_test)

    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (AUC {best['test_auc']})")

    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "model_name": best_name,
            "metrics": {k: v for k, v in best.items() if k != "pipeline"},
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )
    print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
