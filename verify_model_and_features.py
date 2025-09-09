import pandas as pd
import joblib

try:
    print("📦 Loading model_xgb.pkl...")
    model = joblib.load("model_xgb.pkl")
    print("✅ Model loaded successfully!")

    print("\n📑 Loading model_features.csv...")
    features_df = pd.read_csv("model_features.csv")
    print("✅ Feature list loaded!")
    print(f"Number of features: {features_df.shape[1]}")
    print("Sample feature columns:", list(features_df.columns[:5]), "...")

except Exception as e:
    print("❌ Error:", e)
