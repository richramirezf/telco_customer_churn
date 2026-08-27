import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_loader import load_telco_data
from quality import run_quality_pipeline


NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_pipeline() -> Pipeline:
    """Return a sklearn Pipeline with preprocessing + XGBClassifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    random_state=42,
                    use_label_encoder=False,
                ),
            ),
        ]
    )
    return pipeline


def train(csv_path: str | None = None) -> str:
    """Full training routine: load → clean → train → save. Returns model path."""
    # 1. Load & clean
    df = load_telco_data(csv_path)
    df = run_quality_pipeline(df)

    # 2. Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[train] Train={len(X_train):,}  Test={len(X_test):,}")

    # 4. Build & fit
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    accuracy = pipeline.score(X_test, y_test)
    print(f"[train] Test accuracy: {accuracy:.4f}")

    # 6. Save
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[train] Model saved → {model_path}")
    return str(model_path)


if __name__ == "__main__":
    train()
