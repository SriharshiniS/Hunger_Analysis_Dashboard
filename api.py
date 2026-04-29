from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Optional: load dataset (for country/year prediction)
df = pd.read_csv("data/Final_dataset.csv", low_memory=False)


# ---------------- HEALTH CHECK ----------------
@app.route("/")
def home():
    return jsonify({"message": "API is running ✅"})


# ---------------- PREDICT USING COUNTRY + YEAR ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        country = data.get("country")
        year = data.get("year")

        if not country or not year:
            return jsonify({"error": "Provide country and year"}), 400

        # Get row from dataset
        row = df[(df["country"] == country) & (df["year"] == int(year))]

        if row.empty:
            return jsonify({"error": "No data found"}), 404

        row = row.iloc[0]

        # Keep only numeric columns
        X = row.select_dtypes(include=["float64", "int64"])

        # Convert to DataFrame
        X = pd.DataFrame([X])

        # Match training features
        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale
        scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(scaled)[0]

        return jsonify({
            "country": country,
            "year": year,
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- PREDICT USING CUSTOM INPUT ----------------
@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    try:
        data = request.json

        # Convert input to DataFrame
        X = pd.DataFrame([data])

        # Match features
        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale
        scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(scaled)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)