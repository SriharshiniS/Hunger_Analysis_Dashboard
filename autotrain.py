import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🚀 AUTO TRAIN STARTED")

# ---------------- LOAD ----------------
df = pd.read_csv("data/Final_dataset.csv", low_memory=False)

print("📊 Dataset shape:", df.shape)

# ---------------- FIND TARGET ----------------
possible_targets = [
    "hunger_level",
    "hunger_severity_index",
    "food_insecurity_rate",
    "undernourishment_pct"
]

target = None
for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    raise Exception("❌ No valid target column found!")

print(f"🎯 Using target: {target}")

# ---------------- CLEAN ----------------
df = df.select_dtypes(include=["float64", "int64"])

df = df.dropna(subset=[target])
df = df.fillna(df.median())

# ---------------- SPLIT ----------------
X = df.drop(columns=[target])
y = df[target]

print("📊 Features:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
print("🤖 Training model...")

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ---------------- SAVE ----------------
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ AUTO TRAIN COMPLETE")