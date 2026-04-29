from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import gdown
import os

app = Flask(__name__)

# ---------------- SAFE DOWNLOAD FUNCTION ----------------
def download_file(file_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output_path,
                quiet=False,
                fuzzy=True
            )
            print(f"Downloaded: {output_path}")
        except Exception as e:
            print(f"❌ Download failed for {output_path}: {e}")

# ---------------- FILE IDS ----------------
MODEL_ID = "1itiURw6HcKACdwmCR6vMbdTGyllVgdsV"
DATA_ID = "12V104uIg9VeIauKEoe9G9uP4cTD_dPzl"

MODEL_PATH = "model/model.pkl"
DATA_PATH = "data/Final_dataset.csv"

# ---------------- DOWNLOAD FILES ----------------
download_file(MODEL_ID, MODEL_PATH)
download_file(DATA_ID, DATA_PATH)

# ---------------- LOAD FILES ----------------
print("Loading model and data...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("✅ Model and dataset loaded")

# ---------------- HOME ----------------
@app.route("/")
def home():
    countries = sorted(df["country"].dropna().unique())
    years = sorted(df["year"].dropna().unique())

    return render_template(
        "index.html",
        countries=countries,
        years=years
    )

# ---------------- AUTO-FILL DATA ----------------
@app.route("/get_data", methods=["POST"])
def get_data():
    try:
        data = request.json
        country = data.get("country")
        year = int(data.get("year"))

        row = df[(df["country"] == country) & (df["year"] == year)]

        if row.empty:
            return jsonify({"error": "No data found"})

        row = row.iloc[0]

        result = {}
        for f in feature_names:
            try:
                result[f] = float(row.get(f, 0))
            except:
                result[f] = 0

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Ensure correct feature order
        X = pd.DataFrame(
            [[data.get(f, 0) for f in feature_names]],
            columns=feature_names
        )

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]

        # Convert to readable label
        label_map = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        prediction = label_map.get(int(pred), str(pred))

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- FEATURE IMPORTANCE ----------------
@app.route("/feature_importance")
def feature_importance():
    try:
        fi = pd.read_csv("model/feature_importance.csv").head(10)
        return fi.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- INSIGHTS ----------------
@app.route("/insights")
def insights():
    try:
        fi = pd.read_csv("model/feature_importance.csv").head(5)

        insights = [
            f"{row['feature']} strongly impacts hunger levels"
            for _, row in fi.iterrows()
        ]

        return jsonify(insights)

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- HEALTH CHECK ----------------
@app.route("/health")
def health():
    return jsonify({"status": "running"})

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
