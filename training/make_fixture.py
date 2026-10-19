"""Generate a synthetic dataset shaped like Home Credit's application_train.csv.

Why: the real Kaggle data cannot be redistributed, so CI (GitHub-hosted
runners) trains and tests against this fake-but-realistic sample instead.
The numbers are random; only the column names, types and quirks
(missing values, the 365243 placeholder) match the real file.

Usage:
    python training/make_fixture.py --rows 5000 --out data/fixture/application_train.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_fixture(rows: int, seed: int = 47) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=11.9, sigma=0.5, size=rows).round(0)
    credit = (income * rng.uniform(1.0, 6.0, size=rows)).round(0)
    annuity = (credit * rng.uniform(0.03, 0.08, size=rows)).round(1)
    goods = (credit * rng.uniform(0.8, 1.0, size=rows)).round(0)
    days_birth = -rng.integers(21 * 365, 69 * 365, size=rows)
    days_employed = -rng.integers(0, 40 * 365, size=rows)
    unemployed = rng.random(rows) < 0.18
    days_employed = np.where(unemployed, 365243, days_employed)

    ext1 = rng.beta(2, 2, size=rows)
    ext2 = rng.beta(3, 2, size=rows)
    ext3 = rng.beta(3, 2, size=rows)

    # Planted signal: low external scores and high credit/income -> more default.
    logit = (
        -2.6
        - 5.0 * (ext2 - 0.6)
        - 4.5 * (ext3 - 0.6)
        - 2.0 * (ext1 - 0.5)
        + 0.25 * (credit / income - 3.5)
        + rng.normal(0, 0.6, size=rows)
    )
    target = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100002, 100002 + rows),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], rows, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], rows),
            "FLAG_OWN_CAR": rng.choice(["N", "Y"], rows, p=[0.66, 0.34]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], rows, p=[0.69, 0.31]),
            "CNT_CHILDREN": rng.choice([0, 1, 2, 3], rows, p=[0.7, 0.2, 0.08, 0.02]),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant"],
                rows, p=[0.52, 0.23, 0.18, 0.07],
            ),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
                rows, p=[0.71, 0.24, 0.03, 0.02],
            ),
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage"], rows),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
                rows, p=[0.89, 0.05, 0.04, 0.02],
            ),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": -rng.integers(0, 20000, size=rows).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 7000, size=rows),
            "OWN_CAR_AGE": np.where(rng.random(rows) < 0.66, np.nan, rng.integers(0, 30, rows)),
            "CNT_FAM_MEMBERS": rng.choice([1.0, 2.0, 3.0, 4.0], rows),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], rows, p=[0.1, 0.74, 0.16]),
            "EXT_SOURCE_1": np.where(rng.random(rows) < 0.56, np.nan, ext1),
            "EXT_SOURCE_2": np.where(rng.random(rows) < 0.002, np.nan, ext2),
            "EXT_SOURCE_3": np.where(rng.random(rows) < 0.2, np.nan, ext3),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--out", default="data/fixture/application_train.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_fixture(args.rows).to_csv(out, index=False)
    print(f"Wrote {args.rows} synthetic rows to {out}")


if __name__ == "__main__":
    main()
