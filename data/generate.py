"""Generate a synthetic credit risk dataset with realistic distributions."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2000


def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate(n: int = N) -> pd.DataFrame:
    age = _clamp(RNG.normal(38, 12, n), 18, 75).astype(int)
    employment_years = _clamp(RNG.normal(7, 5, n), 0, 40).astype(int)
    annual_income = _clamp(RNG.normal(55_000, 25_000, n), 12_000, 200_000).astype(int)

    loan_purpose = RNG.choice(
        ["personal", "car", "education", "home_improvement", "medical"],
        n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    loan_amount = _clamp(RNG.normal(15_000, 8_000, n), 1_000, 60_000).astype(int)
    loan_duration_months = RNG.choice([12, 24, 36, 48, 60], n, p=[0.10, 0.25, 0.35, 0.20, 0.10])

    num_credit_lines = _clamp(RNG.poisson(3, n), 0, 15).astype(int)
    credit_history = RNG.choice(["good", "fair", "poor"], n, p=[0.55, 0.30, 0.15])

    debt_to_income = _clamp(
        (loan_amount / loan_duration_months * 12) / annual_income + RNG.normal(0, 0.05, n),
        0.01,
        0.95,
    ).round(3)

    # Risk score: higher = more likely to default
    risk = (
        0.3 * (loan_amount / 60_000)
        + 0.2 * (1 - annual_income / 200_000)
        + 0.15 * (1 - employment_years / 40)
        + 0.15 * (1 - (age - 18) / 57)
        + 0.2 * np.where(credit_history == "poor", 1.0, np.where(credit_history == "fair", 0.4, 0.0))
        + RNG.normal(0, 0.08, n)
    )

    default = (risk > 0.45).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "loan_duration_months": loan_duration_months,
            "loan_purpose": loan_purpose,
            "num_credit_lines": num_credit_lines,
            "credit_history": credit_history,
            "debt_to_income_ratio": debt_to_income,
            "default": default,
        }
    )


if __name__ == "__main__":
    df = generate()
    out = "data/credit_risk.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df["default"].value_counts(normalize=True).round(3).to_string())
