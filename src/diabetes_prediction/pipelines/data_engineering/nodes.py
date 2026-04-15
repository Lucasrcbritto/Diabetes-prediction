import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace zeros with NaN (except Pregnancies and Outcome)
    zero_cols = [col for col in df.columns
                 if df[col].min() == 0 and col not in ["Pregnancies", "Outcome"]]
    for col in zero_cols:
        df[col] = np.where(df[col] == 0, np.nan, df[col])

    # KNN imputation
    na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    rs = RobustScaler()
    dff = pd.DataFrame(rs.fit_transform(df[na_cols]), columns=na_cols)
    dff = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(dff), columns=na_cols)
    df[na_cols] = pd.DataFrame(rs.inverse_transform(dff), columns=na_cols)

    # Outlier capping
    for col in df.columns:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Age category
    df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
    df.loc[df["Age"] >= 50, "NEW_AGE_CAT"] = "senior"

    # BMI bins
    df["NEW_BMI"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                           labels=["Underweight", "Healthy", "Overweight", "Obese"])

    # Glucose category
    df["NEW_GLUCOSE"] = pd.cut(df["Glucose"], bins=[0, 140, 200, 300],
                               labels=["Normal", "Prediabetes", "Diabetes"])

    # Insulin score
    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(
        lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

    # Interaction features
    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    # Uppercase columns
    df.columns = [col.upper() for col in df.columns]

    return df