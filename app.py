from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import gdown
import os

app = Flask(__name__)

# ---------------- DOWNLOAD MODEL ----------------
MODEL_ID = "1itiURw6HcKACdwmCR6vMbdTGyllVgdsV"
MODEL_PATH = "model/model.pkl"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}",
                   MODEL_PATH,
                   quiet=False,
                   fuzzy=True)

# ---------------- DOWNLOAD DATASET ----------------
DATA_ID = "12V104uIg9VeIauKEoe9G9uP4cTD_dPzl"
DATA_PATH = "data/Final_dataset.csv"

os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={DATA_ID}",
                   DATA_PATH,
                   quiet=False,
                   fuzzy=True)

# ---------------- LOAD ----------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

df = pd.read_csv(DATA_PATH, low_memory=False)

# ---------------- HOME ----------------
@app.route("/")
def home():
    countries = sorted(df["country"].dropna().unique())
    years = sorted(df["year"].dropna().unique())

    return render_template("index.html",
                           countries=countries,
                           years=years)

# ---------------- AUTO-FILL ----------------
@app.route("/get_data", methods=["POST"])
def get_data():
    data = request.json
    country = data["country"]
    year = int(data["year"])

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
    data = request.json

    X = pd.DataFrame([[data.get(f, 0) for f in feature_names]],
                     columns=feature_names)

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]

    # convert numeric to label
    labels = {0: "Low", 1: "Medium", 2: "High"}
    pred_label = labels.get(int(pred), str(pred))

    return jsonify({
        "prediction": pred_label
    })

# ---------------- FEATURE IMPORTANCE ----------------
@app.route("/feature_importance")
def feature_importance():
    fi = pd.read_csv("model/feature_importance.csv").head(10)
    return fi.to_json(orient="records")

# ---------------- INSIGHTS ----------------
@app.route("/insights")
def insights():
    fi = pd.read_csv("model/feature_importance.csv").head(5)

    insights = []
    for _, row in fi.iterrows():
        insights.append(f"{row['feature']} strongly impacts hunger levels")

    return jsonify(insights)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
