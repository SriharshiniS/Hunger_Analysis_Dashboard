from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------- SAFE LOAD ----------------
def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")
        return None

def safe_load_data():
    try:
        return pd.read_csv("data/Final_dataset.csv", low_memory=False)
    except Exception as e:
        print(f"⚠️ Dataset load failed: {e}")
        return pd.DataFrame()

# Load everything safely
model = safe_load_model("model/model.pkl")
scaler = safe_load_model("model/scaler.pkl")
feature_names = safe_load_model("model/feature_names.pkl")

df = safe_load_data()

# ---------------- HOME ----------------
@app.route("/")
def home():
    if df.empty:
        return "❌ Dataset not found. Check Render setup."

    countries = sorted(df["country"].dropna().unique())
    years = sorted(df["year"].dropna().unique())

    return render_template("index.html",
                           countries=countries,
                           years=years)

# ---------------- AUTO LOAD ----------------
@app.route("/get_data", methods=["POST"])
def get_data():
    if df.empty:
        return jsonify({"error": "Dataset not loaded"})

    data = request.json
    country = data.get("country")
    year = int(data.get("year"))

    row = df[(df["country"] == country) & (df["year"] == year)]

    if row.empty:
        return jsonify({})

    row = row.iloc[0]

    result = {}
    for f in feature_names:
        try:
            result[f] = float(row.get(f, 0))
        except:
            result[f] = 0

    return jsonify(result)

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"})

        data = request.json

        X = pd.DataFrame([[data.get(f, 0) for f in feature_names]],
                         columns=feature_names)

        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]

        confidence = 0
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(X_scaled)[0])

        return jsonify({
            "prediction": str(pred),
            "confidence": round(float(confidence)*100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- FEATURE IMPORTANCE ----------------
@app.route("/feature_importance")
def feature_importance():
    try:
        fi = pd.read_csv("model/feature_importance.csv").head(10)
        return fi.to_json(orient="records")
    except:
        return jsonify([])

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

    except:
        return jsonify([])

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT
    app.run(host="0.0.0.0", port=port)
