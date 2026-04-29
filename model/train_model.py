import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from utils.preprocess import load_data, prepare_features

print("Loading data...")
df = load_data("data/Final_dataset.csv")

print("Preparing features...")
X, y = prepare_features(df, target="hunger_level")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "model/feature_names.pkl")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# ✅ FEATURE IMPORTANCE
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_importance_df.to_csv("model/feature_importance.csv", index=False)

print("✅ Model trained & feature importance saved!")