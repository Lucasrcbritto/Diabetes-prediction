import pandas as pd
from sklearn.preprocessing import RobustScaler


def prepare_inference_data(df: pd.DataFrame, X_train: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    df = df.copy()

    # Convert categorical columns to string and one-hot encode
    cat_cols = [col for col in df.columns
                if df[col].dtype == "O" or str(df[col].dtype) == "category"]
    for col in cat_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align columns with training set
    df = df.reindex(columns=X_train.columns, fill_value=0)

    # Scale ONLY the columns the scaler was fitted on
    scale_cols = scaler.feature_names_in_
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    results = df.copy()
    results["prediction"] = predictions
    results["probability"] = probabilities

    return results