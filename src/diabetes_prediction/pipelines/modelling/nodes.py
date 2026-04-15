import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # Convert ALL categorical/object columns to string so get_dummies works
    cat_cols = [col for col in df.columns
                if df[col].dtype == "O" or str(df[col].dtype) == "category"
                and col != "OUTCOME"]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical columns only
    num_cols = [col for col in df.columns
                if df[col].dtype in ["float64", "int64", "float32", "int32"]
                and col != "OUTCOME"]
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("OUTCOME", axis=1)
    y = df["OUTCOME"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=17
    )

    y_train = pd.DataFrame(y_train).reset_index(drop=True)
    y_test = pd.DataFrame(y_test).reset_index(drop=True)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame):
    y_train = y_train.squeeze()
    params = {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 0.5],
    }
    model = GridSearchCV(XGBClassifier(random_state=46), params, cv=5)
    model.fit(X_train, y_train)
    print(f"Best params: {model.best_params_}")
    return model.best_estimator_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    y_test = y_test.squeeze()
    y_pred = model.predict(X_test)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"AUC:       {roc_auc_score(y_test, y_pred):.4f}")
    return model