from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model files
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")

# Load dataset
df = pd.read_csv("data/Final_dataset.csv", low_memory=False)

@app.route("/")
def home():
    countries = sorted(df["country"].dropna().unique())
    years = sorted(df["year"].dropna().unique())

    return render_template("index.html",
                           countries=countries,
                           years=years)

# ✅ Auto-fill data
@app.route("/get_data", methods=["POST"])
def get_data():
    country = request.json["country"]
    year = request.json["year"]

    row = df[(df["country"] == country) & (df["year"] == int(year))]

    if row.empty:
        return jsonify({})

    row = row.iloc[0]

    data = {}
    for f in feature_names:
        val = row.get(f, 0)
        try:
            data[f] = float(val)
        except:
            data[f] = 0

    return jsonify(data)

# ✅ Prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = pd.DataFrame([[data.get(f, 0) for f in feature_names]],
                     columns=feature_names)

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled).max()

    return jsonify({
        "prediction": str(pred),
        "confidence": round(float(prob)*100, 2)
    })

# ✅ Feature Importance
@app.route("/feature_importance")
def feature_importance():
    fi = pd.read_csv("model/feature_importance.csv").head(10)
    return fi.to_json(orient="records")

# ✅ Insights
@app.route("/insights")
def insights():
    fi = pd.read_csv("model/feature_importance.csv").head(5)

    insights = []
    for _, row in fi.iterrows():
        insights.append(f"{row['feature']} is a key driver of hunger levels")

    return jsonify(insights)

if __name__ == "__main__":
    app.run(debug=True)