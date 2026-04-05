# model.py
# This file trains a Machine Learning model on our student data
# and saves it to disk so we can reuse it in the app

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib  # Used to save/load the trained model

from data import generate_data

def train_model():
    # ── Step 1: Load or generate data ──────────────────────────────
    try:
        df = pd.read_csv("student_data.csv")
        print("📂 Loaded existing dataset")
    except FileNotFoundError:
        print("📊 Generating new dataset...")
        df = generate_data()

    # ── Step 2: Split into features (X) and target (y) ─────────────
    # X = inputs (what we know about the student)
    # y = output (what we want to predict)
    X = df[["study_hours", "attendance", "sleep_hours", "prev_score", "extracurricular"]]
    y = df["final_grade"]

    # ── Step 3: Split into training and testing sets ────────────────
    # 80% of data used to train, 20% used to test how good the model is
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Step 4: Train the model ─────────────────────────────────────
    # RandomForest = many decision trees working together (very accurate)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # This is where "learning" happens

    # ── Step 5: Evaluate the model ──────────────────────────────────
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n📈 Model Performance:")
    print(f"   MAE (Mean Absolute Error): {mae:.2f}  ← avg error in grade points")
    print(f"   R² Score:                  {r2:.4f}  ← 1.0 is perfect, >0.9 is great")

    # ── Step 6: Save the model ──────────────────────────────────────
    joblib.dump(model, "grade_model.pkl")
    print("\n✅ Model trained and saved as grade_model.pkl")

    return model

if __name__ == "__main__":
    train_model()
