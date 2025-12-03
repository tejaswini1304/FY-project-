from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from services.weather_service import WeatherService


from flask import request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# enable CORS for the app (optional but useful)
CORS(app)

weather_service = WeatherService()

# Global variables
feature_scaler = None
target_scaler = None
feature_columns = None
dnn_classifier = None
dnn_regressor = None

MAIN_CROPS = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Tomato', 'Potato', 'Coffee']

CROP_FACTS = {
    'Rice': "Rice is semi-aquatic. It requires standing water or consistently high moisture (>70%).",
    'Wheat': "Wheat is hardy but needs water during 'Crown Root Initiation'. Keep moisture > 45%.",
    'Maize': "Maize is sensitive during silking. Keep above 50% to ensure grain filling.",
    'Cotton': "Cotton needs dry periods to burst bolls. Irrigate only if critical (<35%).",
    'Sugarcane': "Sugarcane is a massive water consumer. Needs consistent moisture (>65%).",
    'Tomato': "Tomatoes need consistency (>60%) to prevent blossom end rot.",
    'Potato': "Shallow roots need frequent moisture (>45%).",
    'Coffee': "Coffee needs a 'Stress Period' (Dry Soil <30%) to flower."
}

# The Rules
CROP_THRESHOLDS = {
    'Rice': 70, 'Sugarcane': 65, 'Tomato': 60, 'Maize': 50,
    'Wheat': 45, 'Potato': 45, 'Cotton': 35, 'Coffee': 30
}

def load_artifacts():
    global feature_scaler, target_scaler, feature_columns, dnn_classifier, dnn_regressor
    path = "models/saved/"
    try:
        if os.path.exists(path):
            feature_scaler = joblib.load(f"{path}feature_scaler.pkl")
            target_scaler = joblib.load(f"{path}target_scaler.pkl")
            feature_columns = joblib.load(f"{path}feature_columns.pkl")
            
            dnn_classifier = tf.keras.models.load_model(f"{path}dnn_classifier.h5", compile=False)
            dnn_regressor = tf.keras.models.load_model(f"{path}dnn_regressor.h5", compile=False)
            print("✅ Artifacts Loaded.")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

load_artifacts()

def get_prediction(input_dict):
    try:
        df_input = pd.DataFrame(input_dict)
        df_input = pd.get_dummies(df_input, columns=["CROP_TYPE", "SOIL_TYPE", "REGION", "WEATHER_CONDITION"])
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)
        
        cols = ["TEMPERATURE", "HUMIDITY", "RAINFALL", "WIND_SPEED", "SOIL_MOISTURE"]
        df_input[cols] = feature_scaler.transform(df_input[cols])
        
        need_prob = float(dnn_classifier.predict(df_input, verbose=0)[0][0])
        water_scaled = dnn_regressor.predict(df_input, verbose=0)
        water_amount = float(target_scaler.inverse_transform(water_scaled)[0][0])
        water_amount = max(0.0, water_amount)
        
        return need_prob, water_amount
    except Exception as e:
        print(e); return 0.0, 0.0

@app.route("/")
def home():
    options = {
        "CROP_TYPE": MAIN_CROPS,
        "SOIL_TYPE": ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky'],
        "REGION": ['North', 'East', 'West', 'South', 'Central'],
        "WEATHER_CONDITION": ['Sunny', 'Rainy', 'Cloudy', 'Windy']
    }
    return render_template("index.html", options=options)

@app.route("/get_weather", methods=["POST"])
def get_weather():
    city = request.json.get("city")
    data = weather_service.get_weather_data(city)
    return jsonify(data)

@app.route("/predict_forecast", methods=["POST"])
def predict_forecast():
    try:
        data = request.json
        city = data.get('city')
        base = data.get('base_inputs')
        forecasts = weather_service.get_forecast_data(city)
        if "error" in forecasts: return jsonify(forecasts)
        
        preds = []
        crop = base['crop_type']
        threshold = CROP_THRESHOLDS.get(crop, 50)

        for day in forecasts:
            curr_moist = float(base['soil_moisture']) - 5
            day_in = {
                "CROP_TYPE": [crop], "SOIL_TYPE": [base['soil_type']], "REGION": [base['region']],
                "TEMPERATURE": [day['temperature']], "HUMIDITY": [day['humidity']],
                "RAINFALL": [day['rainfall']], "WIND_SPEED": [day['wind_speed']],
                "WEATHER_CONDITION": [day['weather_condition']], "SOIL_MOISTURE": [curr_moist]
            }
            prob, amount = get_prediction(day_in)
            
            # Forecast Logic
            needs = True
            if day['rainfall'] > 10: needs = False; amount = 0.0
            if curr_moist > threshold: needs = False; amount = 0.0
            
            preds.append({
                "day": day['day_name'], "date": day['date_short'],
                "condition": day['condition_desc'], "temp": day['temperature'],
                "needs_water": needs, "amount": round(amount, 1)
            })
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    global dnn_classifier
    if dnn_classifier is None: load_artifacts()

    try:
        data = request.json
        crop = data.get("crop_type")
        moist = float(data.get("soil_moisture", 0))
        rain = float(data.get("rainfall", 0))
        temp = float(data.get("temperature", 0))
        
        input_data = {
            "CROP_TYPE": [crop], "SOIL_TYPE": [data.get("soil_type")], "REGION": [data.get("region")],
            "TEMPERATURE": [temp], "HUMIDITY": [float(data.get("humidity", 0))],
            "RAINFALL": [rain], "WIND_SPEED": [float(data.get("wind_speed", 0))],
            "WEATHER_CONDITION": [data.get("weather_condition")], "SOIL_MOISTURE": [moist]
        }

        prob, water_amount = get_prediction(input_data)
        needs_water = prob > 0.5
        
        advice = "Conditions are optimal."
        threshold = CROP_THRESHOLDS.get(crop, 50)
        
        # --- SMART LOGIC ---
        # 1. Force Irrigation (Critical Dryness)
        if moist < threshold:
            needs_water = True
            # If model predicted 0 in error, rescue it
            if water_amount < 1.0: water_amount = 3.5 
            advice = f"Moisture ({moist}%) is below {crop} limit ({threshold}%). Irrigation Required."

        # 2. Force Stop (Heavy Rain)
        elif rain > 15:
            needs_water = False
            water_amount = 0.0
            advice = f"Rainfall detected ({rain}mm). Irrigation Skipped."

        # 3. Force Stop (Healthy Soil)
        elif moist > threshold:
            needs_water = False
            water_amount = 0.0
            advice = f"Soil Moisture ({moist}%) is sufficient for {crop}."

        temp_impact = min(100, max(0, (temp - 15) * 4))
        moist_impact = min(100, max(0, (100 - moist)))
        rain_impact = min(100, max(0, rain * 5))

        return jsonify({
            "needs_water": bool(needs_water),
            "confidence": round(prob, 4),
            "water_amount": round(water_amount, 2),
            "advice": advice,
            "crop_fact": CROP_FACTS.get(crop, ""),
            "impacts": {"temp": int(temp_impact), "moisture": int(moist_impact), "rain": int(rain_impact)}
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

from flask import request, jsonify
from translator import tr

@app.route("/translate_texts", methods=["POST"])
def translate_texts():
    data = request.json
    lang = data.get("lang")
    texts = data.get("texts", {})

    translated = {}
    for key, value in texts.items():
        translated[key] = tr(value, lang)

    return jsonify(translated)

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"reply": "", "error": "Empty message"}), 400

    system_prompt = (
        "You are AgriAssist, a friendly agriculture advisor for Indian farmers. "
        "Use simple language, short sentences, and bullet points. "
        "Give practical farming steps. "
        "If needed, ask one clarifying question. "
        "Do NOT give dangerous chemical pesticide advice."
    )

    # Build message history
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # UPDATED MODEL
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply, "error": None})

    except Exception as e:
        print("Groq chatbot error:", str(e))
        return jsonify({"reply": "", "error": str(e)}), 500


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)