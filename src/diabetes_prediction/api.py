from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Diabetes Prediction API")

# Load model and scaler
with open("data/06_models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/06_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("data/05_model_input/X_train.parquet", "rb") as f:
    X_train = pd.read_parquet(f)


class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
    df.loc[df["Age"] >= 50, "NEW_AGE_CAT"] = "senior"

    df["NEW_BMI"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                           labels=["Underweight", "Healthy", "Overweight", "Obese"])

    df["NEW_GLUCOSE"] = pd.cut(df["Glucose"], bins=[0, 140, 200, 300],
                               labels=["Normal", "Prediabetes", "Diabetes"])

    df["NEW_INSULIN_SCORE"] = df["Insulin"].apply(
        lambda x: "Normal" if 16 <= x <= 166 else "Abnormal")

    df["NEW_GLUCOSE_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    df.columns = [col.upper() for col in df.columns]
    return df


@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running!"}


@app.post("/predict")
def predict(patient: PatientData):
    # Convert input to DataFrame
    df = pd.DataFrame([patient.model_dump()])

    # Feature engineering
    df = feature_engineering(df)

    # Encode categorical columns
    cat_cols = [col for col in df.columns
                if df[col].dtype == "O" or str(df[col].dtype) == "category"]
    for col in cat_cols:
        df[col] = df[col].astype(str)
    df = pd.get_dummies(df, drop_first=True)

    # Align with training columns
    df = df.reindex(columns=X_train.columns, fill_value=0)

    # Scale
    scale_cols = scaler.feature_names_in_
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Predict
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "diagnosis": "Diabetic" if prediction == 1 else "Not Diabetic",
        "probability": round(probability, 4)
    }