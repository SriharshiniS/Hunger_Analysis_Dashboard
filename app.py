from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---------------- LOAD FILES ----------------
import gdown
import os

file_id = "12V104uIg9VeIauKEoe9G9uP4cTD_dPzl"
url = f"https://drive.google.com/uc?id={file_id}"

output = "data/Final_dataset.csv"

os.makedirs("data", exist_ok=True)

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

df = pd.read_csv(output, low_memory=False)


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


# ---------------- AUTO LOAD DATA ----------------
@app.route("/get_data", methods=["POST"])
def get_data():
    try:
        data = request.json
        country = data.get("country")
        year = int(data.get("year"))

        row = df[(df["country"] == country) & (df["year"] == year)]

        if row.empty:
            return jsonify({"error": "No data found"}), 404

        row = row.iloc[0]

        result = {}
        for f in feature_names:
            val = row.get(f, 0)

            # Clean value safely
            try:
                result[f] = float(val)
            except:
                result[f] = 0

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Ensure ALL features are present
        X = pd.DataFrame([[data.get(f, 0) for f in feature_names]],
                         columns=feature_names)

        # Handle NaN / invalid
        X = X.fillna(0)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)[0]

        # Probability (safe check)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled).max()
            confidence = round(float(prob) * 100, 2)
        else:
            confidence = "N/A"

        return jsonify({
            "prediction": str(pred),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- FEATURE IMPORTANCE ----------------
@app.route("/feature_importance")
def feature_importance():
    try:
        fi_path = os.path.join(BASE_DIR, "model/feature_importance.csv")
        fi = pd.read_csv(fi_path).head(10)

        return fi.to_json(orient="records")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- INSIGHTS ----------------
@app.route("/insights")
def insights():
    try:
        fi_path = os.path.join(BASE_DIR, "model/feature_importance.csv")
        fi = pd.read_csv(fi_path).head(5)

        insights = [
            f"{row['feature']} strongly influences hunger prediction"
            for _, row in fi.iterrows()
        ]

        return jsonify(insights)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 🔥 REQUIRED FOR RENDER
    app.run(host="0.0.0.0", port=port)
