import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

RANDOM_STATE = 42
CSV_PATH = "fertilizer_recommendation_dataset.csv"
MODEL_OUT = "fertilizer_models.pkl"
ENCODER_OUT = "label_encoder.pkl"

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Inspect expected columns in this dataset:
    expected = ["PH", "Nitrogen", "Phosphorous", "Potassium", "Moisture", "Fertilizer"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing}")

    # Keep only relevant columns (drop others like Temperature, Rainfall, Soil, Remark)
    df = df[expected].copy()

    # Rename to convenient names (optional)
    df.rename(columns={"PH": "pH", "Phosphorous": "Phosphorus"}, inplace=True)

    # Convert numeric columns to numeric (coerce errors)
    numeric_cols = ["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing numeric values with median
    for c in numeric_cols:
        if df[c].isna().any():
            med = df[c].median()
            df[c].fillna(med, inplace=True)

    # Drop rows missing target
    df = df.dropna(subset=["Fertilizer"])
    df = df.reset_index(drop=True)
    return df

def balance_by_upsampling(df, target_col):
    # Upsample minority classes to the largest class size
    counts = df[target_col].value_counts()
    max_count = counts.max()

    frames = []
    for cls, cnt in counts.items():
        cls_df = df[df[target_col] == cls]
        if cnt < max_count:
            cls_upsampled = resample(cls_df,
                                     replace=True,
                                     n_samples=max_count,
                                     random_state=RANDOM_STATE)
            frames.append(cls_upsampled)
        else:
            frames.append(cls_df)
    balanced = pd.concat(frames).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return balanced

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    print("Loading dataset...")
    df = load_and_clean(CSV_PATH)
    print("Dataset loaded. Class distribution before balancing:")
    print(df["Fertilizer"].value_counts())

    # Balance dataset (optional — comment out if you prefer class_weight only)
    df_balanced = balance_by_upsampling(df, "Fertilizer")
    print("\nAfter upsampling, class distribution:")
    print(df_balanced["Fertilizer"].value_counts())

    # Features and target
    X = df_balanced[["pH", "Nitrogen", "Phosphorus", "Potassium", "Moisture"]].copy()
    y = df_balanced["Fertilizer"].copy()

    # Encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split (stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}\n")

    print("Classification report (test set):")
    target_names = le.inverse_transform(sorted(set(y_train)))
    # Build mapping for readable report
    # Note: classification_report expects labels indexed 0..n-1, so we can call with target_names=le.classes_
    print(classification_report(y_test, y_pred_test, target_names=le.classes_, zero_division=0))

    # Save model and encoder
    joblib.dump(model, MODEL_OUT)
    joblib.dump(le, ENCODER_OUT)
    print(f"✔ Model saved to: {MODEL_OUT}")
    print(f"✔ Label encoder saved to: {ENCODER_OUT}")

if __name__ == "__main__":
    main()
