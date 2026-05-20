import streamlit as st
import numpy as np
import joblib
import time
import random
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

model = joblib.load("fertilizer_models.pkl")
tips = {
    "English": """
    ### Best Practices:
    1. **Soil Preparation**
       - Test soil before planting
       - Maintain proper pH levels
       - Ensure good drainage
       - Add organic matter if needed

    2. **Fertilizer Application**
       - Follow recommended dosages
       - Apply at right growth stages
       - Use organic fertilizers when possible
       - Consider split applications

    3. **Water Management**
       - Monitor soil moisture
       - Implement drip irrigation
       - Avoid over-watering
       - Consider weather forecasts

    4. **Sustainable Practices**
       - Practice crop rotation
       - Use cover crops
       - Implement integrated pest management
       - Maintain soil health
    """,

    "Kannada": """
    ### ಉತ್ತಮ ಅಭ್ಯಾಸಗಳು:
    1. **ಮಣ್ಣಿನ ತಯಾರಿ**
       - ನೆಡುವ ಮೊದಲು ಮಣ್ಣನ್ನು ಪರೀಕ್ಷಿಸಿ
       - ಸರಿಯಾದ pH ಮಟ್ಟವನ್ನು ಕಾಪಾಡಿ
       - ಉತ್ತಮ ನೀರುಹೊರಹೋಗುವ ವ್ಯವಸ್ಥೆ ಇರಲಿ
       - ಅಗತ್ಯವಿದ್ದರೆ ಸಾವಯವ ಪದಾರ್ಥವನ್ನು ಸೇರಿಸಿ

    2. **ರಸಗೊಬ್ಬರದ ಬಳಕೆ**
       - ಶಿಫಾರಸು ಮಾಡಿದ ಪ್ರಮಾಣವನ್ನು ಅನುಸರಿಸಿ
       - ಸರಿಯಾದ ಬೆಳವಣಿಗೆಯ ಹಂತಗಳಲ್ಲಿ ಅನ್ವಯಿಸಿ
       - ಸಾಧ್ಯವಾದಾಗ ಸಾವಯವ ರಸಗೊಬ್ಬರಗಳನ್ನು ಬಳಸಿ
       - ವಿಭಜಿತ ಅನ್ವಯಿಕೆಯನ್ನು ಪರಿಗಣಿಸಿ

    3. **ನೀರಿನ ನಿರ್ವಹಣೆ**
       - ಮಣ್ಣಿನ ತೇವಾಂಶವನ್ನು ಗಮನಿಸಿ
       - ಡ್ರಿಪ್ ನೀರಾವರಿ ವಿಧಾನವನ್ನು ಅಳವಡಿಸಿ
       - ಅತಿಯಾದ ನೀರಾವರಿಯನ್ನು ತಪ್ಪಿಸಿ
       - ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಯನ್ನು ಪರಿಗಣಿಸಿ

    4. **ಸ್ಥಿರಾಭಿವೃದ್ಧಿ ಅಭ್ಯಾಸಗಳು**
       - ಬೆಳೆ ಪರಿವರ್ತನೆ ಅಭ್ಯಾಸವನ್ನು ಅನುಸರಿಸಿ
       - ಕವರ್ ಬೆಳೆಗಳನ್ನು ಬಳಸಿ
       - ಸಂಯೋಜಿತ ಕೀಟ ನಿರ್ವಹಣೆಯನ್ನು ಅಳವಡಿಸಿ
       - ಮಣ್ಣಿನ ಆರೋಗ್ಯವನ್ನು ಕಾಪಾಡಿ
    """,

    "Hindi": """
    ### सर्वोत्तम अभ्यास:
    1. **मिट्टी की तैयारी**
       - रोपण से पहले मिट्टी का परीक्षण करें
       - उचित pH स्तर बनाए रखें
       - अच्छी जल निकासी सुनिश्चित करें
       - आवश्यकता होने पर जैविक पदार्थ जोड़ें

    2. **उर्वरक का उपयोग**
       - अनुशंसित मात्रा का पालन करें
       - सही वृद्धि चरणों में लगाएँ
       - संभव हो तो जैविक उर्वरक का उपयोग करें
       - विभाजित अनुप्रयोग पर विचार करें

    3. **जल प्रबंधन**
       - मिट्टी की नमी की निगरानी करें
       - ड्रिप सिंचाई लागू करें
       - अधिक सिंचाई से बचें
       - मौसम पूर्वानुमान पर विचार करें

    4. **सतत अभ्यास**
       - फसल चक्र का अभ्यास करें
       - कवर फसलें उपयोग करें
       - एकीकृत कीट प्रबंधन लागू करें
       - मिट्टी के स्वास्थ्य को बनाए रखें
    """
}
texts = {
    "English": {
        "title": "🌱 GreenGrow AI - Smart Soil Analysis System",
        "about": "|AI-powered Soil Analysis • Weather Forecast • Fertilizer Guidance • Multi-language Support|",
        "upload": "Upload Soil Image",
        "analyze": "Analyze Soil",
        "results": "🧪 Analysis Results",
        "ph": "Soil pH Level",
        "moisture": "Moisture Content",
        "nitrogen": "Nitrogen Level",
        "phosphorus": "Phosphorus Level",
        "potassium": "Potassium Level",
        "recommend": "Recommended Crop",
        "fertilizer": "Suggested Fertilizer",
        "no_file": "Please upload a soil image to proceed.",
        "success": "Analysis completed successfully!",
        "soil_analysis": "🧪 Soil Analysis",
        "weather_forecast": "🌦️ Weather Forecast",
        "historical_data": "📚 Historical Data",
        "enter_details": "📊 Enter the Soil Details",
        "tab1": "Soil Analysis",
        "tab2": "Weather Forecast",
        "tab3": "Historical Data",
        "title1": "### Historical Statistics",
        "yield": "Average Yield",
        "rainfall": "Average Rainfall",
        "temperature": "Average Temperature",
        "fertilizer": "Average Fertilizer Usage",
        "moisture": "Average Soil Moisture",
        "ph": "Average Soil pH",
        "price": "Average Market Price",
        "labor": "Average Labor Cost",
        "current": " Current Soil Parameters:",
        "micro": "Micronutrients:",
        "fert_title": "🌱 Fertilizer Recommendations for",
        "req": "Required Nutrients (kg/ha):",
        "rec": "Recommended Fertilizers:",
        "health": "🌍 Soil Health Status:",
        "ph_opt": "pH is optimal",
        "ph_adj": "pH needs adjustment",
        "n_low": "Nitrogen level is low",
        "n_suf": "Nitrogen level is sufficient",
        "p_low": "Phosphorus level is low",
        "p_suf": "Phosphorus level is sufficient",
        "k_low": "Potassium level is low",
        "k_suf": "Potassium level is sufficient",
        "m_opt": "Moisture level is optimal",
        "m_adj": "Moisture needs adjustment",
        "o_good": "Organic matter is good",
        "o_low": "Organic matter is low",
        "fetching_weather": "Fetching weather data...",
        "temp_rainfall_chart": "Temperature & Rainfall",
        "humidity_wind_chart": "Humidity & Wind Speed",
        "temperature": "Temperature (°C)",
        "rainfall": "Rainfall (mm)",
        "humidity": "Humidity (%)",
        "wind_speed": "Wind Speed (km/h)",
        "from": "from",
        "pressure": "Pressure",
        "visibility": "Visibility",
        "condition": "Condition",
        "get_weather_button": "🔄 Get Weather Report",
        "current_weather": "Current Weather Conditions",
        "weather_alerts": "🌾 Farming Weather Alerts",
        "high_temp_alert": "⚠️ High Temperature Alert — Avoid fertilizer application during peak hours.",
        "low_temp_alert": "⚠️ Low Temperature Alert — Delay fertilizer application until temperatures rise.",
        "heavy_rain_alert": "⚠️ Heavy Rainfall Alert — Postpone fertilizer application to prevent runoff.",
        "dry_spell_alert": "⚠️ Dry Spell Alert — Increase irrigation frequency.",
        "strong_wind_alert": "⚠️ Strong Wind Alert — Postpone spraying operations.",
        "calm_wind_alert": "ℹ️ Calm Wind Conditions — Ideal for spraying and foliar applications.",
        "high_humidity_alert": "⚠️ High Humidity Alert — Increased risk of fungal diseases.",
        "low_humidity_alert": "⚠️ Low Humidity Alert — Increase irrigation and use shade nets.",
        "heat_stress_alert": "⚠️ Heat Stress Alert — Increase irrigation and monitor for wilting.",
        "storm_alert": "⚠️ Storm Alert — Secure farm equipment and postpone operations.",
        "favorable_weather": "✅ Weather is favorable for farming.",
        "weather_fetch_error": "Failed to fetch weather data. Please try again later.",
        "tab4": "Farmer Benefits",

    },
    "Hindi": {
        "title": "🌱 ग्रीनग्रो AI - स्मार्ट मिट्टी विश्लेषण प्रणाली",
        "about": "|एआई-संचालित मृदा विश्लेषण • मौसम पूर्वानुमान • उर्वरक मार्गदर्शन • बहुभाषी समर्थन|",
        "upload": "मिट्टी की छवि अपलोड करें",
        "analyze": "मिट्टी का विश्लेषण करें",
        "results": "🧪 विश्लेषण परिणाम",
        "ph": "मिट्टी का पीएच स्तर",
        "moisture": "नमी की मात्रा",
        "nitrogen": "नाइट्रोजन स्तर",
        "phosphorus": "फास्फोरस स्तर",
        "potassium": "पोटेशियम स्तर",
        "recommend": "अनुशंसित फसल",
        "fertilizer": "सुझाया गया उर्वरक",
        "no_file": "कृपया मिट्टी की छवि अपलोड करें।",
        "success": "विश्लेषण सफलतापूर्वक पूरा हुआ!",
        "soil_analysis": "🧪 मिट्टी विश्लेषण",
        "weather_forecast": "🌦️ मौसम का पूर्वानुमान",
        "historical_data": "📚 ऐतिहासिक डेटा",
        "enter_details": "📊 मिट्टी के विवरण दर्ज करें",
        "tab1": "मिट्टी विश्लेषण",
        "tab2": "मौसम का पूर्वानुमान",
        "tab3": "ऐतिहासिक डेटा",
        "title1": "### ऐतिहासिक सांख्यिकी",
        "yield": "औसत उत्पादन",
        "rainfall": "औसत वर्षा",
        "temperature": "औसत तापमान",
        "fertilizer": "औसत उर्वरक उपयोग",
        "moisture": "औसत मिट्टी नमी",
        "ph": "औसत मिट्टी pH",
        "price": "औसत बाजार मूल्य",
        "labor": "औसत श्रम लागत",
        "current": "वर्तमान मिट्टी के पैरामीटर:",
        "micro": "सूक्ष्म पोषक तत्व:",
        "fert_title": "🌱 धान के लिए उर्वरक सिफारिशें:",
        "req": "आवश्यक पोषक तत्व (किग्रा/हे):",
        "rec": "अनुशंसित उर्वरक:",
        "health": "🌍 मिट्टी की स्वास्थ्य स्थिति:",
        "ph_opt": "pH सामान्य है",
        "ph_adj": "pH समायोजन की आवश्यकता है",
        "n_low": "नाइट्रोजन स्तर कम है",
        "n_suf": "नाइट्रोजन स्तर पर्याप्त है",
        "p_low": "फास्फोरस स्तर कम है",
        "p_suf": "फास्फोरस स्तर पर्याप्त है",
        "k_low": "पोटेशियम स्तर कम है",
        "k_suf": "पोटेशियम स्तर पर्याप्त है",
        "m_opt": "नमी स्तर सामान्य है",
        "m_adj": "नमी समायोजन की आवश्यकता है",
        "o_good": "जैविक पदार्थ अच्छा है",
        "o_low": "जैविक पदार्थ कम है",
        "fetching_weather": "मौसम डेटा प्राप्त किया जा रहा है...",
        "temp_rainfall_chart": "तापमान और वर्षा",
        "humidity_wind_chart": "आर्द्रता और पवन गति",
        "temperature": "तापमान (°C)",
        "rainfall": "वर्षा (mm)",
        "humidity": "आर्द्रता (%)",
        "wind_speed": "पवन गति (किमी/घंटा)",
        "from": "की दिशा से",
        "pressure": "दबाव",
        "visibility": "दृश्यता",
        "condition": "मौसम की स्थिति",
        "current_weather": "वर्तमान मौसम की स्थिति",
        "weather_alerts": "🌾 खेती के लिए मौसम चेतावनियाँ",
        "high_temp_alert": "⚠️ उच्च तापमान चेतावनी — चरम समय (10 बजे से 4 बजे) में उर्वरक का उपयोग न करें।",
        "low_temp_alert": "⚠️ कम तापमान चेतावनी — तापमान बढ़ने तक उर्वरक का प्रयोग न करें।",
        "heavy_rain_alert": "⚠️ भारी वर्षा चेतावनी — उर्वरक का प्रयोग रोक दें ताकि बहाव न हो।",
        "dry_spell_alert": "⚠️ शुष्क अवधि चेतावनी — सिंचाई की आवृत्ति बढ़ाएँ।",
        "strong_wind_alert": "⚠️ तेज़ हवा चेतावनी — छिड़काव कार्य स्थगित करें।",
        "calm_wind_alert": "ℹ️ शांत हवा की स्थिति — छिड़काव और पत्तों पर स्प्रे के लिए उपयुक्त समय।",
        "high_humidity_alert": "⚠️ उच्च आर्द्रता चेतावनी — फफूंदी रोगों का खतरा बढ़ जाता है।",
        "low_humidity_alert": "⚠️ कम आर्द्रता चेतावनी — सिंचाई बढ़ाएँ और शेड नेट का उपयोग करें।",
        "heat_stress_alert": "⚠️ ताप तनाव चेतावनी — सिंचाई बढ़ाएँ और मुरझाने की निगरानी करें।",
        "storm_alert": "⚠️ तूफान चेतावनी — उपकरण सुरक्षित रखें और कार्य स्थगित करें।",
        "favorable_weather": "✅ खेती के लिए मौसम अनुकूल है।",
        "weather_fetch_error": "मौसम डेटा प्राप्त करने में विफल। कृपया बाद में पुनः प्रयास करें।",
        "get_weather_button": "🔄 मौसम रिपोर्ट प्राप्त करें",
        "tab4": "किसान लाभ",

    },
    "Kannada": {
        "title": "🌱 ಗ್ರೀನ್‌ಗ್ರೋ AI - ಸ್ಮಾರ್ಟ್ ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣಾ ವ್ಯವಸ್ಥೆ",
        "about": "|ಎಐ ಆಧಾರಿತ ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ • ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ • ರಸಗೊಬ್ಬರ ಮಾರ್ಗದರ್ಶನ • ಬಹುಭಾಷಾ ಬೆಂಬಲ|",
        "upload": "ಮಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "analyze": "ಮಣ್ಣನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "results": "🧪 ವಿಶ್ಲೇಷಣಾ ಫಲಿತಾಂಶಗಳು",
        "ph": "ಮಣ್ಣಿನ pH ಮಟ್ಟ",
        "moisture": "ತೇವಾಂಶ ಪ್ರಮಾಣ",
        "nitrogen": "ನೈಟ್ರೋಜನ್ ಮಟ್ಟ",
        "phosphorus": "ಫಾಸ್ಫರಸ್ ಮಟ್ಟ",
        "potassium": "ಪೊಟ್ಯಾಸಿಯಮ್ ಮಟ್ಟ",
        "recommend": "ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ",
        "fertilizer": "ಸೂಚಿಸಿದ ರಸಗೊಬ್ಬರ",
        "no_file": "ದಯವಿಟ್ಟು ಮಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "success": "ವಿಶ್ಲೇಷಣೆ ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಂಡಿದೆ!",
        "Farming Tips": "ಕೃಷಿ ಸಲಹೆಗಳು",
        "soil_analysis": "🧪 ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ",
        "weather_forecast": "🌦️ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
        "historical_data": "📚 ಐತಿಹಾಸಿಕ ಮಾಹಿತಿ",
        "enter_details": "📊 ಮಣ್ಣಿನ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ",
        "tab1": "ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ",
        "tab2": "ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
        "tab3": "ಐತಿಹಾಸಿಕ ಮಾಹಿತಿ",
        "title1": "### ಐತಿಹಾಸಿಕ ಅಂಕಿಅಂಶಗಳು",
        "yield": "ಸರಾಸರಿ ಉತ್ಪಾದನೆ",
        "rainfall": "ಸರಾಸರಿ ಮಳೆ",
        "temperature": "ಸರಾಸರಿ ತಾಪಮಾನ",
        "fertilizer": "ಸರಾಸರಿ ರಸಗೊಬ್ಬರ ಬಳಕೆ",
        "moisture": "ಸರಾಸರಿ ಮಣ್ಣಿನ ತೇವಾಂಶ",
        "ph": "ಸರಾಸರಿ ಮಣ್ಣಿನ pH",
        "price": "ಸರಾಸರಿ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ",
        "labor": "ಸರಾಸರಿ ಕಾರ್ಮಿಕ ವೆಚ್ಚ",
        "current": "ಪ್ರಸ್ತುತ ಮಣ್ಣಿನ ಮಾನಗಳು:",
        "micro": "ಕ್ಷುದ್ರಪೋಷಕಾಂಶಗಳು:",
        "fert_title": "🌱 ಅಕ್ಕಿಗೆ ರಸಗೊಬ್ಬರ ಶಿಫಾರಸುಗಳು:",
        "req": "ಅವಶ್ಯಕ ಪೋಷಕಾಂಶಗಳು (ಕೆಜಿ/ಹೆ):",
        "rec": "ಶಿಫಾರಸು ಮಾಡಿದ ರಸಗೊಬ್ಬರಗಳು:",
        "health": "🌍 ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಸ್ಥಿತಿ:",
        "ph_opt": "pH ಸರಿಯಾಗಿದೆ",
        "ph_adj": "pH ತಿದ್ದುಪಡಿ ಅಗತ್ಯವಿದೆ",
        "n_low": "ನೈಟ್ರೋಜನ್ ಮಟ್ಟ ಕಡಿಮೆ",
        "n_suf": "ನೈಟ್ರೋಜನ್ ಮಟ್ಟ ಸಮರ್ಪಕವಾಗಿದೆ",
        "p_low": "ಫಾಸ್ಫರಸ್ ಮಟ್ಟ ಕಡಿಮೆ",
        "p_suf": "ಫಾಸ್ಫರಸ್ ಮಟ್ಟ ಸಮರ್ಪಕವಾಗಿದೆ",
        "k_low": "ಪೋಟ್ಯಾಸಿಯಮ್ ಮಟ್ಟ ಕಡಿಮೆ",
        "k_suf": "ಪೋಟ್ಯಾಸಿಯಮ್ ಮಟ್ಟ ಸಮರ್ಪಕವಾಗಿದೆ",
        "m_opt": "ತೇವಾಂಶ ಸರಿಯಾಗಿದೆ",
        "m_adj": "ತೇವಾಂಶ ತಿದ್ದುಪಡಿ ಅಗತ್ಯವಿದೆ",
        "o_good": "ಸಾವಯವ ಪದಾರ್ಥ ಒಳ್ಳೆಯದು",
        "o_low": "ಸಾವಯವ ಪದಾರ್ಥ ಕಡಿಮೆ",
        "fetching_weather": "ಹವಾಮಾನ ಮಾಹಿತಿಯನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ...",
        "temp_rainfall_chart": "ತಾಪಮಾನ ಮತ್ತು ಮಳೆ",
        "humidity_wind_chart": "ಆದ್ರತೆ ಮತ್ತು ಗಾಳಿಯ ವೇಗ",
        "temperature": "ತಾಪಮಾನ (°C)",
        "rainfall": "ಮಳೆ (ಮಿಮೀ)",
        "humidity": "ಆದ್ರತೆ (%)",
        "wind_speed": "ಗಾಳಿಯ ವೇಗ (ಕಿ.ಮೀ/ಗಂ)",
        "from": "ಇಂದ",
        "pressure": "ಒತ್ತಡ",
        "visibility": "ದೃಶ್ಯಮಾನತೆ",
        "condition": "ಹವಾಮಾನ ಸ್ಥಿತಿ",
        "current_weather": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಸ್ಥಿತಿ",
        "weather_alerts": "🌾 ಕೃಷಿಗೆ ಸಂಬಂಧಿಸಿದ ಹವಾಮಾನ ಎಚ್ಚರಿಕೆಗಳು",
        "high_temp_alert": "⚠️ ಹೆಚ್ಚು ತಾಪಮಾನ ಎಚ್ಚರಿಕೆ — ಮಧ್ಯಾಹ್ನ ಸಮಯದಲ್ಲಿ (10AM–4PM) ರಸಗೊಬ್ಬರ ನೀಡಬೇಡಿ.",
        "low_temp_alert": "⚠️ ಕಡಿಮೆ ತಾಪಮಾನ ಎಚ್ಚರಿಕೆ — ತಾಪಮಾನ ಏರಿದ ನಂತರ ಮಾತ್ರ ರಸಗೊಬ್ಬರ ನೀಡಿ.",
        "heavy_rain_alert": "⚠️ ಭಾರಿ ಮಳೆಯ ಎಚ್ಚರಿಕೆ — ರಸಗೊಬ್ಬರ ನೀಡುವುದನ್ನು ಮುಂದೂಡಿ, ನೀರಿನ ಹರಿವು ತಪ್ಪಿಸಿ.",
        "dry_spell_alert": "⚠️ ಒಣಹವಾಮಾನ ಎಚ್ಚರಿಕೆ — ನೀರಾವರಿ ಪ್ರಮಾಣ ಹೆಚ್ಚಿಸಿ.",
        "strong_wind_alert": "⚠️ ಬಲವಾದ ಗಾಳಿ ಎಚ್ಚರಿಕೆ — ಸಿಂಪಡಣೆ ಕಾರ್ಯವನ್ನು ಮುಂದೂಡಿ.",
        "calm_wind_alert": "ℹ️ ನಿಶ್ಚಲ ಗಾಳಿ ಪರಿಸ್ಥಿತಿ — ಸಿಂಪಡಣೆ ಮತ್ತು ಎಲೆ ಸಿಂಪಡಣೆಗಾಗಿ ಉತ್ತಮ ಸಮಯ.",
        "high_humidity_alert": "⚠️ ಹೆಚ್ಚು ಆದ್ರತೆ ಎಚ್ಚರಿಕೆ — ಹುಳ ರೋಗಗಳ ಅಪಾಯ ಹೆಚ್ಚಿದೆ.",
        "low_humidity_alert": "⚠️ ಕಡಿಮೆ ಆದ್ರತೆ ಎಚ್ಚರಿಕೆ — ನೀರಾವರಿ ಹೆಚ್ಚಿಸಿ ಮತ್ತು ಷೇಡ್ ನೆಟ್ ಬಳಸಿ.",
        "heat_stress_alert": "⚠️ ಉಷ್ಣ ಒತ್ತಡ ಎಚ್ಚರಿಕೆ — ನೀರಾವರಿ ಹೆಚ್ಚಿಸಿ ಮತ್ತು ಒಣಗುವಿಕೆಯನ್ನು ಗಮನಿಸಿ.",
        "storm_alert": "⚠️ ಬಿರುಗಾಳಿ ಎಚ್ಚರಿಕೆ — ಉಪಕರಣಗಳನ್ನು ಸುರಕ್ಷಿತಪಡಿಸಿ ಮತ್ತು ಕಾರ್ಯವನ್ನು ಮುಂದೂಡಿ.",
        "favorable_weather": "✅ ಕೃಷಿಗೆ ಹವಾಮಾನ ಅನುಕೂಲಕರವಾಗಿದೆ.",
        "weather_fetch_error": "ಹವಾಮಾನ ಮಾಹಿತಿಯನ್ನು ಪಡೆಯಲು ವಿಫಲವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "get_weather_button": "🔄 ಹವಾಮಾನ ವರದಿ ಪಡೆಯಿರಿ",
        "tab4": "ರೈತರಿಗೆ ಲಾಭಗಳು",
    }
}
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* ======================================================
   GLOBAL STYLING
====================================================== */
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif !important;
}

/* Smooth App Background */
.stApp {
    background: linear-gradient(145deg, #c8ffd4, #eaffea, #ffffff);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

/* ======================================================
   GLASSMORPHIC CARD
====================================================== */
.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.35);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.20);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    margin-bottom: 20px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.card:hover {
    transform: scale(1.02);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
}

/* ======================================================
   HEADER
====================================================== */
h1 {
    color: #006b3c;
    font-size: 42px !important;
    font-weight: 700;
    text-shadow: 2px 2px 10px rgba(0,255,100,0.25);
    margin-bottom: 5px;
}

.green-tagline {
    text-align:center;
    font-size:18px;
    color:#055d2e;
    text-shadow: 0px 0px 8px rgba(0,255,80,0.2);
    font-weight:500;
}

/* ======================================================
   PREMIUM TABS
====================================================== */
.stTabs [data-baseweb="tab"] {
    font-size: 19px !important;
    padding: 10px 20px;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.4);
    backdrop-filter: blur(8px);
    margin-right: 8px;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.6);
}

/* Active tab glow */
.stTabs [aria-selected="true"] {
    background: #00c853 !important;
    color: white !important;
    box-shadow: 0px 0px 12px #00ff73aa;
}

/* ======================================================
   SIDEBAR
====================================================== */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(12px);
    border-right: 2px solid rgba(255,255,255,0.4);
}

/* Sidebar headers */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #006b3c !important;
}

/* ======================================================
   BUTTONS
====================================================== */
.stButton>button {
    border-radius: 14px;
    background: linear-gradient(135deg, #00c853, #009624);
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    border: none;
    box-shadow: 0px 4px 20px rgba(0,255,50,0.3);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #00e676, #00c853);
    box-shadow: 0px 6px 28px rgba(0,255,80,0.5);
}

/* ======================================================
   DATAFRAME
====================================================== */
[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: 0px 6px 30px rgba(0,0,0,0.2);
}

/* ======================================================
   SECTION TITLES
====================================================== */
.section-title {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #00c853, #007f3b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ======================================================
   FLOATING GLOW EFFECT (OPTIONAL)
====================================================== */
.glow {
    animation: glowPulse 2s ease-in-out infinite;
}

@keyframes glowPulse {
    0% { text-shadow: 0px 0px 6px rgba(0,255,80,0.3); }
    50% { text-shadow: 0px 0px 14px rgba(0,255,80,0.7); }
    100% { text-shadow: 0px 0px 6px rgba(0,255,80,0.3); }
}

</style>
""", unsafe_allow_html=True)


farmer_benefits = {
    "English": {
        "title": "🌾 Benefits for Farmers",
        "header": "### ✅ Why This System Helps Farmers",
        "points": [
            "Accurate fertilizer recommendations — reduced wastage and improved yield.",
            "Weather-aware guidance — avoids fertilizer loss during unexpected rains.",
            "Smart decisions based on soil nutrients (N, P, K) and crop type.",
            "Cost savings — prevents unnecessary fertilizer use.",
            "Increased productivity — balanced nutrients lead to healthier crops.",
            "Environmental protection — reduces soil and water pollution.",
            "Easy to use — simple interface for all farmers.",
            "Supports multiple languages — English, Hindi, and Kannada for farmer convenience."
        ],
        "summary_title": "### 📊 Benefits (Summary)",
        "summary": [
            "25–40% reduction in fertilizer wastage.",
            "10–20% increase in crop yield.",
            "Better soil health.",
            "Improved long-term sustainability."
        ],
        "footer": "This system empowers farmers to make smart, data-driven agricultural decisions!"
    },

    "Hindi": {
        "title": "🌾 किसानों के लिए लाभ",
        "header": "### ✅ यह प्रणाली किसानों की कैसे मदद करती है?",
        "points": [
            "सटीक उर्वरक सिफारिशें — बर्बादी कम, उत्पादन अधिक।",
            "मौसम आधारित सलाह — अप्रत्याशित बारिश में उर्वरक का नुकसान नहीं।",
            "मिट्टी (N, P, K) और फसल के आधार पर स्मार्ट निर्णय।",
            "लागत में बचत — अनावश्यक उर्वरक उपयोग से बचाव।",
            "उत्पादन में वृद्धि — संतुलित पोषण से स्वस्थ फसल।",
            "पर्यावरण की सुरक्षा — मिट्टी और जल प्रदूषण में कमी।",
            "उपयोग में आसान — सभी किसानों के लिए सरल इंटरफ़ेस।",
            "बहुभाषी समर्थन — अंग्रेजी, हिंदी और कन्नड़ में उपलब्ध।"
        ],
        "summary_title": "### 📊 लाभ (सारांश)",
        "summary": [
            "25–40% उर्वरक की बर्बादी में कमी।",
            "10–20% उत्पादन में वृद्धि।",
            "बेहतर मिट्टी स्वास्थ्य।",
            "लंबे समय तक टिकाऊ खेती।"
        ],
        "footer": "यह प्रणाली किसानों को डेटा आधारित, समझदारी भरे कृषि निर्णय लेने में सक्षम बनाती है!"
    },

    "Kannada": {
        "title": "🌾 ರೈತರಿಗೆ ಲಾಭಗಳು",
        "header": "### ✅ ಈ ವ್ಯವಸ್ಥೆ ರೈತರಿಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ?",
        "points": [
            "ನಿಖರ ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು — ವ್ಯರ್ಥ ಕಡಿಮೆ, ಉತ್ಪಾದನೆ ಹೆಚ್ಚು.",
            "ಹವಾಮಾನ ಆಧಾರಿತ ಸಲಹೆ — ಅನಿರೀಕ್ಷಿತ ಮಳೆಯಿಂದ ನಷ್ಟ ತಪ್ಪಿಸಬಹುದು.",
            "ಮಣ್ಣಿನ (N, P, K) ಮತ್ತು ಬೆಳೆಯ ಆಧಾರದ ಮೇಲೆ ಬುದ್ಧಿವಂತ ನಿರ್ಧಾರ.",
            "ವೆಚ್ಚದಲ್ಲಿ ಉಳಿತಾಯ — ಅನಗತ್ಯ ರಸಗೊಬ್ಬರ ಬಳಕೆ ತಪ್ಪುತ್ತದೆ.",
            "ಉತ್ಪಾದನೆ ಹೆಚ್ಚಳ — ಸಮತೋಲನ ಪೋಷಕಾಂಶಗಳಿಂದ ಆರೋಗ್ಯಕರ ಬೆಳೆ.",
            "ಪರಿಸರ ರಕ್ಷಣೆ — ಮಣ್ಣು ಮತ್ತು ನೀರಿನ ಮಾಲಿನ್ಯ ಕಡಿಮೆ.",
            "ಬಳಕೆ ಸುಲಭ — ಎಲ್ಲ ರೈತರಿಗೆ ಅನುಕೂಲಕರ.",
            "ಬಹುಭಾಷಾ ಬೆಂಬಲ — ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ ಮತ್ತು ಕನ್ನಡದಲ್ಲಿ ಲಭ್ಯ."
        ],
        "summary_title": "### 📊 ಲಾಭಗಳು (ಸಾರಾಂಶ)",
        "summary": [
            "25–40% ರಸಗೊಬ್ಬರ ವ್ಯರ್ಥ ಕಡಿತ.",
            "10–20% ಉತ್ಪಾದನೆ ಹೆಚ್ಚಳ.",
            "ಉತ್ತಮ ಮಣ್ಣು ಆರೋಗ್ಯ.",
            "ದೀರ್ಘಾವಧಿಯ ಸುಸ್ಥಿರ ಕೃಷಿ."
        ],
        "footer": "ಈ ವ್ಯವಸ್ಥೆ ರೈತರಿಗೆ ಡೇಟಾ ಆಧಾರಿತ ಬುದ್ಧಿವಂತ ಕೃಷಿ ನಿರ್ಧಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ!"
    }
}

st.set_page_config(page_title="🌏GreenGrow AI ", layout="wide")
language = st.sidebar.selectbox(
    "Choose Language / भाषा चुनें / ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ",
    ("English", "Hindi", "Kannada")
)


selected = texts[language]

st.markdown(f"""
<div style="text-align:center; padding:8px 0; margin-bottom:10px;">
    <h1>{selected['title']}</h1>
    <p style="font-size:18px; color:#2e7d32;">{selected['about']}</p>
</div>
""", unsafe_allow_html=True)


# ---------------------- 🌐 IoT Data Integration ----------------------
def get_iot_data():
    """Fetch latest IoT sensor readings from ThingSpeak"""
    url = "https://api.thingspeak.com/channels/3138119/feeds.json?api_key=P6SCVY4THF4V7ZKC&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()['feeds'][0]
        readings = {
            "pH": float(data['field1']),
            "nitrogen": float(data['field2']),
            "phosphorus": float(data['field3']),
            "potassium": float(data['field4']),
            "moisture": float(data['field5']),
            "temperature": float(data['field6']),
            "humidity": float(data['field7']),
        }
        return readings
    except Exception as e:
        st.error(f"⚠️ Error fetching IoT data: {e}")
        return None
# --------------------------------------------------------------------

st.sidebar.title("🔑 API Configuration")

api_key = st.sidebar.text_input(
    "OpenWeatherMap API Key",
    value="7459717f774319dc60fc0031557a8147",
    type="password",
    help="Get your free API key from https://openweathermap.org/api"
)

if st.sidebar.button("Test API Connection"):
    with st.sidebar:
        with st.spinner("Testing connection..."):
            try:
                test_params = {
                    "lat": 20.5937,
                    "lon": 78.9629,
                    "appid": api_key,
                    "units": "metric"
                }

                response = requests.get(BASE_URL, params=test_params)

                if response.status_code == 200:
                    st.success("✅ API connection successful!")
                else:
                    data = response.json()

                    if response.status_code == 401:
                        st.error("❌ Invalid API key")
                    else:
                        st.error(f"❌ Error: {data.get('message', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
st.sidebar.markdown("---")

st.sidebar.link_button(
    "⬇ Download Desktop App",
    "https://github.com/Rahul-pr-0503/Fertilizer_Recommendation/releases/download/v1.0/GreenGrow.AI.Setup.1.0.0.zip"
)

def get_real_weather_data(latitude, longitude):
    try:
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            forecast = []
            for item in data['list']:
                forecast.append({
                    "date": datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M'),
                    "temperature": round(item['main']['temp'], 1),
                    "humidity": item['main']['humidity'],
                    "rainfall": item.get('rain', {}).get('3h', 0),
                    "wind_speed": round(item['wind']['speed'] * 3.6, 1),
                    "wind_direction": get_wind_direction(item['wind']['deg']),
                    "cloud_cover": item['clouds']['all'],
                    "pressure": round(item['main']['pressure'], 1),
                    "visibility": round(item['visibility'] / 1000, 1),
                    "description": item['weather'][0]['description'],
                    "icon": item['weather'][0]['icon']
                })
            return forecast
        else:
            if response.status_code == 401:
                st.error("""
                ❌ Invalid API key. Please check:
                1. Did you copy the entire API key?
                2. Did you wait 2 hours after activation?
                3. Is your account email verified?
                4. Try clicking 'Test API Connection' in the sidebar
                """)
            else:
                st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_wind_direction(degrees):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(degrees / (360. / len(directions))) % len(directions)
    return directions[index]


CROPS = {
    "Rice": {
        "N": 120, "P": 60, "K": 60, "pH": (5.5, 6.5),
        "growth_stages": ["Seedling", "Tillering", "Panicle Initiation", "Flowering", "Grain Filling"],
        "water_requirement": "High",
        "temperature_range": (20, 35),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Basmati", "Non-Basmati", "Hybrid"],
        "yield_potential": "4-6 tons/ha"
    },
    "Wheat": {
        "N": 100, "P": 50, "K": 50, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Tillering", "Stem Elongation", "Heading", "Ripening"],
        "water_requirement": "Medium",
        "temperature_range": (15, 25),
        "season": ["Rabi"],
        "varieties": ["Durum", "Bread Wheat", "Emmer"],
        "yield_potential": "3-5 tons/ha"
    },
    "Maize": {
        "N": 120, "P": 60, "K": 60, "pH": (5.8, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Tasseling", "Silking", "Maturity"],
        "water_requirement": "Medium",
        "temperature_range": (18, 32),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Sweet Corn", "Field Corn", "Popcorn"],
        "yield_potential": "5-8 tons/ha"
    },
    "Soybean": {
        "N": 20, "P": 60, "K": 80, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Development", "Maturity"],
        "water_requirement": "Medium",
        "temperature_range": (20, 30),
        "season": ["Kharif"],
        "varieties": ["Black", "Yellow", "Green"],
        "yield_potential": "2-3 tons/ha"
    },
    "Cotton": {
        "N": 100, "P": 50, "K": 50, "pH": (5.8, 6.5),
        "growth_stages": ["Germination", "Vegetative", "Square Formation", "Flowering", "Boll Development"],
        "water_requirement": "Medium",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Upland", "Pima", "Egyptian"],
        "yield_potential": "2-3 bales/ha"
    },
    "Potato": {
        "N": 120, "P": 60, "K": 120, "pH": (5.0, 6.0),
        "growth_stages": ["Sprouting", "Vegetative", "Tuber Initiation", "Tuber Bulking", "Maturity"],
        "water_requirement": "High",
        "temperature_range": (15, 25),
        "season": ["Rabi"],
        "varieties": ["Russet", "Red", "White"],
        "yield_potential": "20-30 tons/ha"
    },
    "Tomato": {
        "N": 100, "P": 50, "K": 150, "pH": (5.5, 6.8),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Fruit Setting", "Harvesting"],
        "water_requirement": "Medium",
        "temperature_range": (20, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Cherry", "Beefsteak", "Roma"],
        "yield_potential": "40-60 tons/ha"
    },
    "Sugarcane": {
        "N": 200, "P": 100, "K": 200, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Tillering", "Grand Growth", "Maturity"],
        "water_requirement": "High",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Early", "Mid", "Late"],
        "yield_potential": "80-100 tons/ha"
    },
    "Millets": {
        "N": 60, "P": 30, "K": 30, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Grain Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Pearl", "Finger", "Foxtail"],
        "yield_potential": "1.5-2.5 tons/ha"
    },
    "Pulses": {
        "N": 20, "P": 40, "K": 20, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 30),
        "season": ["Rabi"],
        "varieties": ["Chickpea", "Lentil", "Pigeon Pea"],
        "yield_potential": "1-2 tons/ha"
    },
    "Oilseeds": {
        "N": 40, "P": 20, "K": 20, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Mustard", "Sunflower", "Groundnut"],
        "yield_potential": "1.5-2.5 tons/ha"
    },
    "Vegetables": {
        "N": 80, "P": 40, "K": 60, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Fruit Setting"],
        "water_requirement": "High",
        "temperature_range": (15, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Leafy", "Root", "Fruit"],
        "yield_potential": "20-30 tons/ha"
    }
}

FERTILIZERS = {
    "Nitrogen": {
        "Urea": {"N": 46, "brands": ["IFFCO", "KRIBHCO", "Nagarjuna", "Chambal", "Tata", "Coromandel"]},
        "Ammonium Nitrate": {"N": 34, "brands": ["Coromandel", "Zuari", "GSFC", "RCF"]},
        "Ammonium Sulfate": {"N": 21, "brands": ["RCF", "GSFC", "IFFCO"]},
        "Calcium Ammonium Nitrate": {"N": 26, "brands": ["Yara", "Haifa", "ICL"]}
    },
    "Phosphorus": {
        "DAP": {"P": 46, "brands": ["IFFCO", "Coromandel", "Zuari", "Paradeep", "RCF"]},
        "SSP": {"P": 16, "brands": ["RCF", "GSFC", "IFFCO", "Coromandel"]},
        "Rock Phosphate": {"P": 30, "brands": ["Paradeep", "Jhamarkotra", "RSMML"]},
        "NPK Complex": {"P": 20, "brands": ["IFFCO", "Coromandel", "Zuari"]}
    },
    "Potassium": {
        "MOP": {"K": 60, "brands": ["IPL", "Zuari", "Coromandel", "IFFCO"]},
        "SOP": {"K": 50, "brands": ["IFFCO", "KRIBHCO", "Yara"]},
        "Potassium Nitrate": {"K": 44, "brands": ["Yara", "Haifa", "ICL"]}
    },
    "Micronutrients": {
        "Zinc Sulfate": {"Zn": 21, "brands": ["Coromandel", "Zuari", "IFFCO"]},
        "Boron": {"B": 11, "brands": ["Yara", "Haifa", "ICL"]},
        "Iron Chelate": {"Fe": 12, "brands": ["Yara", "Haifa", "ICL"]}
    }
}


def get_sensor_readings():
    readings = {
        "pH": round(random.uniform(5.0, 8.0), 1),
        "nitrogen": random.randint(20, 100),
        "phosphorus": random.randint(20, 100),
        "potassium": random.randint(20, 100),
        "moisture": round(random.uniform(30.0, 70.0), 1),
        "temperature": round(random.uniform(15.0, 35.0), 1),
        "humidity": random.randint(40, 90),
        "organic_matter": round(random.uniform(1.0, 5.0), 1),
        "ec": round(random.uniform(0.5, 3.0), 1),
        "soil_type": random.choice(["Sandy", "Loamy", "Clayey"]),
        "micronutrients": {
            "zinc": round(random.uniform(0.5, 2.0), 1),
            "iron": round(random.uniform(2.0, 10.0), 1),
            "manganese": round(random.uniform(1.0, 5.0), 1),
            "copper": round(random.uniform(0.2, 1.0), 1),
            "boron": round(random.uniform(0.2, 1.0), 1)
        }
    }
    return readings


BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

def get_weather_data(lat, lon, api_key):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        forecast = []
        for item in data["list"]:
            forecast.append({
                "date": datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M"),
                "temperature": round(item["main"]["temp"], 1),   # ✅ renamed
                "humidity": item["main"]["humidity"],
                "rainfall": item.get("rain", {}).get("3h", 0),   # ✅ renamed for clarity
                "wind_speed": round(item["wind"]["speed"] * 3.6, 1),
                "wind_direction": get_wind_direction(item["wind"]["deg"]),
                "pressure": item["main"]["pressure"],
                "visibility": round(item.get("visibility", 0) / 1000, 1),
                "description": item["weather"][0]["description"]
            })
        return forecast
    else:
        return None


def calculate_fertilizer_requirements(soil_readings, crop_type):
    crop_needs = CROPS[crop_type]
    recommendations = {
        "N": max(0, crop_needs["N"] - soil_readings["nitrogen"]),
        "P": max(0, crop_needs["P"] - soil_readings["phosphorus"]),
        "K": max(0, crop_needs["K"] - soil_readings["potassium"])
    }

    fertilizer_details = {
        "Nitrogen": {
            "Urea": round(recommendations["N"] / 0.46, 1),
            "Ammonium Nitrate": round(recommendations["N"] / 0.34, 1),
            "Calcium Ammonium Nitrate": round(recommendations["N"] / 0.26, 1)
        },
        "Phosphorus": {
            "DAP": round(recommendations["P"] / 0.46, 1),
            "SSP": round(recommendations["P"] / 0.16, 1),
            "NPK Complex": round(recommendations["P"] / 0.20, 1)
        },
        "Potassium": {
            "MOP": round(recommendations["K"] / 0.60, 1),
            "SOP": round(recommendations["K"] / 0.50, 1),
            "Potassium Nitrate": round(recommendations["K"] / 0.44, 1)
        }
    }
    
    return recommendations, fertilizer_details


def generate_historical_data(crop_type):
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
    data = {
        'Date': dates,
        'Yield': [random.uniform(2.0, 4.0) for _ in range(12)],
        'Rainfall': [random.uniform(0, 200) for _ in range(12)],
        'Temperature': [random.uniform(15, 35) for _ in range(12)],
        'Fertilizer_Used': [random.uniform(100, 300) for _ in range(12)],
        'Soil_Moisture': [random.uniform(30, 70) for _ in range(12)],
        'Soil_pH': [random.uniform(5.0, 8.0) for _ in range(12)],
        'Pest_Incidence': [random.uniform(0, 100) for _ in range(12)],
        'Disease_Incidence': [random.uniform(0, 100) for _ in range(12)],
        'Market_Price': [random.uniform(1000, 5000) for _ in range(12)],
        'Labor_Cost': [random.uniform(500, 2000) for _ in range(12)]
    }
    return pd.DataFrame(data)
st.sidebar.title("📡 Input Mode")
input_mode = st.sidebar.radio("Select Input Source:", ["Manual", "IoT Sensor"])

st.sidebar.title("🌱 Crop Selection")
selected_crop = st.sidebar.selectbox("Select Crop Type", list(CROPS.keys()))

st.sidebar.title("📍 Location")
latitude = st.sidebar.number_input("Latitude", value=20.5937, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.9629, format="%.4f")


st.sidebar.markdown(f"""
### {selected_crop} Requirements:
- **Nitrogen (N)**: {CROPS[selected_crop]['N']} kg/ha
- **Phosphorus (P)**: {CROPS[selected_crop]['P']} kg/ha
- **Potassium (K)**: {CROPS[selected_crop]['K']} kg/ha
- **Optimal pH**: {CROPS[selected_crop]['pH'][0]} - {CROPS[selected_crop]['pH'][1]}
- **Water Requirement**: {CROPS[selected_crop]['water_requirement']}
- **Temperature Range**: {CROPS[selected_crop]['temperature_range'][0]}°C - {CROPS[selected_crop]['temperature_range'][1]}°C
- **Growing Season**: {', '.join(CROPS[selected_crop]['season'])}
- **Varieties**: {', '.join(CROPS[selected_crop]['varieties'])}
- **Yield Potential**: {CROPS[selected_crop]['yield_potential']}
""")


tab1, tab2, tab3 , tab4 = st.tabs([
    texts[language]["tab1"],
    texts[language]["tab2"],
    texts[language]["tab3"],
    texts[language]["tab4"]
])

with tab1:
    readings = None  # ✅ ensure readings exists before use
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(texts[language]["enter_details"])

        # ----------- IoT Mode Integration ------------
        if input_mode == "IoT Sensor":
            st.info("📡 Fetching live data from IoT sensors via ThingSpeak...")

            # Fetch data from your ThingSpeak channel
            url = "https://api.thingspeak.com/channels/3138119/feeds.json?api_key=P6SCVY4THF4V7ZKC&results=1"

            try:
                response = requests.get(url, timeout=5)
                data = response.json()['feeds'][0]

                readings = {
                    "pH": float(data['field1']),
                    "nitrogen": float(data['field2']),
                    "phosphorus": float(data['field3']),
                    "potassium": float(data['field4']),
                    "moisture": float(data['field5']),
                    "temperature": float(data['field6']),
                    "humidity": float(data['field7']),
                }

                st.success("✅ Live IoT data received successfully!")
                st.write(readings)

                # Add default placeholders for other soil properties
                readings["organic_matter"] = 2.5
                readings["ec"] = 1.5
                readings["soil_type"] = "Loamy"
                readings["micronutrients"] = {
                    "zinc": 1.0,
                    "iron": 5.0,
                    "manganese": 3.0,
                    "copper": 0.5,
                    "boron": 0.5
                }

            except Exception as e:
                st.error(f"⚠️ Failed to fetch IoT data: {e}")
                st.stop()
        # ---------------------------------------------

        # Manual Mode (user input form)
        else:
            with st.form("user_input_form"):
                pH = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
                nitrogen = st.number_input("Nitrogen (ppm)", min_value=0, max_value=300, value=50, step=1)
                phosphorus = st.number_input("Phosphorus (ppm)", min_value=0, max_value=300, value=40, step=1)
                potassium = st.number_input("Potassium (ppm)", min_value=0, max_value=300, value=60, step=1)
                moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
                temperature = st.number_input("Soil Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
                humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60, step=1)
                organic_matter = st.number_input("Organic Matter (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
                ec = st.number_input("EC (dS/m)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
                soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clayey"])
                zinc = st.number_input("Zinc (ppm)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
                iron = st.number_input("Iron (ppm)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
                manganese = st.number_input("Manganese (ppm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
                copper = st.number_input("Copper (ppm)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
                boron = st.number_input("Boron (ppm)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
                submitted = st.form_submit_button("🔍 Analyse Soil")

            if submitted:
                readings = {
                    "pH": pH,
                    "nitrogen": nitrogen,
                    "phosphorus": phosphorus,
                    "potassium": potassium,
                    "moisture": moisture,
                    "temperature": temperature,
                    "humidity": humidity,
                    "organic_matter": organic_matter,
                    "ec": ec,
                    "soil_type": soil_type,
                    "micronutrients": {
                        "zinc": zinc,
                        "iron": iron,
                        "manganese": manganese,
                        "copper": copper,
                        "boron": boron
                    }
                }

    # ✅ Ensure readings exist before plotting
    if not readings:
        st.warning("⚠️ No sensor or manual data available yet.")
      

    # -------- Plot Gauges --------
    if readings is None:
        st.warning("⚠️ No soil data available yet.")
    else:

        # --- PREMIUM GREENGLASS GAUGE FUNCTION ---
        def fancy_gauge(value, title, min_val, max_val, color):
            return go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': f"<b>{title}</b>", 'font': {'size': 20, 'color': '#004d25'}},
                number={'font': {'size': 34, 'color': color}},
                gauge={
                    'axis': {
                        'range': [min_val, max_val],
                        'tickwidth': 1.2,
                        'tickcolor': "#5c5c5c"
                    },
                    'bar': {
                        'color': color,
                        'thickness': 0.3
                    },
                    'bgcolor': "rgba(255,255,255,0.4)",
                    'borderwidth': 2,
                    'bordercolor': color,
                    'steps': [
                        {'range': [min_val, (min_val + max_val)/2], 'color': 'rgba(0,0,0,0.05)'},
                        {'range': [(min_val + max_val)/2, max_val], 'color': 'rgba(0,0,0,0.10)'}
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 5},
                        'thickness': 0.8,
                        'value': value
                    }
                }
            )


        # --- APPLY TO ALL GAUGES ---
        fig_soil = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )

        fig_soil.add_trace(fancy_gauge(readings['pH'], "pH Level", 5, 8, "#00c853"), row=1, col=1)
        fig_soil.add_trace(fancy_gauge(readings['moisture'], "Moisture %", 0, 100, "#2e7d32"), row=1, col=2)
        fig_soil.add_trace(fancy_gauge(readings['organic_matter'], "Organic Matter %", 0, 5, "#b71c1c"), row=1, col=3)

        fig_soil.add_trace(fancy_gauge(readings['nitrogen'], "Nitrogen (ppm)", 0, 100, "#2962ff"), row=2, col=1)
        fig_soil.add_trace(fancy_gauge(readings['phosphorus'], "Phosphorus (ppm)", 0, 100, "#aa00ff"), row=2, col=2)
        fig_soil.add_trace(fancy_gauge(readings['potassium'], "Potassium (ppm)", 0, 100, "#ff9100"), row=2, col=3)

        fig_soil.update_layout(
            height=430,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_soil, use_container_width=True)


        # -------- Show Current Soil Parameters --------
        st.markdown(f"""
            {texts[language]['current']}
            - **pH Level**: {readings['pH']}
            - **Nitrogen (N)**: {readings['nitrogen']} ppm
            - **Phosphorus (P)**: {readings['phosphorus']} ppm
            - **Potassium (K)**: {readings['potassium']} ppm
            - **Moisture**: {readings['moisture']}%
            - **Temperature**: {readings['temperature']}°C
            - **Humidity**: {readings['humidity']}%
            - **Organic Matter**: {readings['organic_matter']}%
            - **EC**: {readings['ec']} dS/m
            - **Soil Type**: {readings['soil_type']}

            {texts[language]['micro']}
            - **Zinc**: {readings['micronutrients']['zinc']} ppm
            - **Iron**: {readings['micronutrients']['iron']} ppm
            - **Manganese**: {readings['micronutrients']['manganese']} ppm
            - **Copper**: {readings['micronutrients']['copper']} ppm
            - **Boron**: {readings['micronutrients']['boron']} ppm
            """)

        # -------- Fertilizer Recommendation --------
        recommendations, fertilizer_details = calculate_fertilizer_requirements(readings, selected_crop)

        st.success(f"""
            {texts[language]['fert_title']} {selected_crop}:

            **{texts[language]['req']}**
            - Nitrogen (N): {recommendations['N']:.1f}
            - Phosphorus (P): {recommendations['P']:.1f}
            - Potassium (K): {recommendations['K']:.1f}

            **{texts[language]['rec']}**
            - Urea: {fertilizer_details['Nitrogen']['Urea']:.1f} kg/ha
            - DAP: {fertilizer_details['Phosphorus']['DAP']:.1f} kg/ha
            - MOP: {fertilizer_details['Potassium']['MOP']:.1f} kg/ha
            """)

        # -------- Soil Health Status --------
        st.info(f"""
            {texts[language]['health']}
            - {texts[language]['ph_opt'] if CROPS[selected_crop]['pH'][0] <= readings['pH'] <= CROPS[selected_crop]['pH'][1] else texts[language]['ph_adj']}
            - {texts[language]['n_suf'] if readings['nitrogen'] >= CROPS[selected_crop]['N'] * 0.8 else texts[language]['n_low']}
            - {texts[language]['p_suf'] if readings['phosphorus'] >= CROPS[selected_crop]['P'] * 0.8 else texts[language]['p_low']}
            - {texts[language]['k_suf'] if readings['potassium'] >= CROPS[selected_crop]['K'] * 0.8 else texts[language]['k_low']}
            - {texts[language]['m_opt'] if 40 <= readings['moisture'] <= 60 else texts[language]['m_adj']}
            - {texts[language]['o_good'] if readings['organic_matter'] >= 2.0 else texts[language]['o_low']}
            """)

    with col2:
        st.subheader("📝 Farming Tips")
        st.markdown(tips[language])

with tab2:
    st.subheader(f"{texts[language]['tab2']}")

    if not api_key:
        st.warning("⚠️ Please enter your OpenWeatherMap API key in the sidebar to view live weather data.")
        st.stop()
    else:
        if st.button(texts[language]['get_weather_button']):
            with st.spinner("Fetching live weather data..."):
                forecast = get_weather_data(latitude, longitude, api_key)

            if forecast:
                df = pd.DataFrame(forecast)

                # --- Weather Graphs ---
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=(
                        "<b>🌡️ Temperature & 🌧️ Rainfall</b>",
                        "<b>💧 Humidity & 🍃 Wind Speed</b>"
                    )
                )

                # --- TEMP LINE (Gradient Glow Style) ---
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], 
                        y=df["temperature"], 
                        name="Temperature (°C)",
                        line=dict(
                            color="#ff1744",
                            width=4
                        ),
                        hovertemplate="Temperature: %{y}°C<br>%{x}"
                    ),
                    row=1, col=1
                )

                # --- RAINFALL BAR (Glass Style) ---
                fig.add_trace(
                    go.Bar(
                        x=df["date"], 
                        y=df["rainfall"],
                        name="Rainfall (mm)",
                        marker=dict(
                            color="rgba(0, 150, 255, 0.5)",
                            line=dict(color="#0091ea", width=1.5)
                        ),
                        hovertemplate="Rainfall: %{y}mm<br>%{x}"
                    ),
                    row=1, col=1
                )

                # --- HUMIDITY LINE ---
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], 
                        y=df["humidity"], 
                        name="Humidity (%)",
                        line=dict(
                            color="#2962ff",
                            width=4
                        ),
                        hovertemplate="Humidity: %{y}%<br>%{x}"
                    ),
                    row=2, col=1
                )

                # --- WIND SPEED LINE ---
                fig.add_trace(
                    go.Scatter(
                        x=df["date"], 
                        y=df["wind_speed"], 
                        name="Wind Speed (km/h)",
                        line=dict(
                            color="#00c853",
                            width=4
                        ),
                        hovertemplate="Wind: %{y} km/h<br>%{x}"
                    ),
                    row=2, col=1
                )

                # --- PREMIUM LAYOUT ---
                fig.update_layout(
                    height=650,
                    plot_bgcolor="rgba(255,255,255,0.45)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.25,
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(255,255,255,0.4)",
                        bordercolor="rgba(0,0,0,0.1)",
                        borderwidth=1,
                        font=dict(size=13)
                    ),
                )

                # --- Grid Lines (soft + rounded) ---
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=0.4,
                    gridcolor="rgba(0,0,0,0.1)",
                    tickangle=45
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=0.4,
                    gridcolor="rgba(0,0,0,0.1)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Data Table ---
                st.dataframe(df)

                # --- Current Weather Summary ---
                current_weather = df.iloc[0]
                st.markdown(f"""
                    ### {texts[language]['current_weather']}
                    - **{texts[language]['temperature']}**: {current_weather['temperature']}°C  
                    - **{texts[language]['humidity']}**: {current_weather['humidity']}%  
                    - **{texts[language]['wind_speed']}**: {current_weather['wind_speed']} km/h {texts[language]['from']} {current_weather['wind_direction']}  
                    - **{texts[language]['pressure']}**: {current_weather['pressure']} hPa  
                    - **{texts[language]['visibility']}**: {current_weather['visibility']} km  
                    - **{texts[language]['condition']}**: {current_weather['description'].title()}  
                    """)

                st.subheader(texts[language]['weather_alerts'])
                alerts_triggered = False

                # 🌡️ Temperature Alerts
                if current_weather['temperature'] > 35:
                    alerts_triggered = True
                    st.warning(texts[language]['high_temp_alert'])
                elif current_weather['temperature'] < 10:
                    alerts_triggered = True
                    st.warning(texts[language]['low_temp_alert'])

                    # 🌧️ Rainfall Alerts
                if current_weather['rainfall'] > 10:
                    alerts_triggered = True
                    st.warning(texts[language]['heavy_rain_alert'])
                elif current_weather['rainfall'] == 0 and df['rainfall'].sum() < 5:
                    alerts_triggered = True
                    st.warning(texts[language]['dry_spell_alert'])

                    # 🌬️ Wind Alerts
                if current_weather['wind_speed'] > 30:
                    alerts_triggered = True
                    st.warning(texts[language]['strong_wind_alert'])
                elif current_weather['wind_speed'] < 5:
                    alerts_triggered = True
                    st.info(texts[language]['calm_wind_alert'])

                    # 💧 Humidity Alerts
                if current_weather['humidity'] > 80:
                    alerts_triggered = True
                    st.warning(texts[language]['high_humidity_alert'])
                elif current_weather['humidity'] < 40:
                    alerts_triggered = True
                    st.warning(texts[language]['low_humidity_alert'])

                    # ☀️ Heat Stress
                if current_weather['temperature'] > 30 and current_weather['humidity'] > 70:
                    alerts_triggered = True
                    st.warning(texts[language]['heat_stress_alert'])

                    # 🌩️ Storm Alert
                if current_weather['rainfall'] > 5 and current_weather['wind_speed'] > 20:
                    alerts_triggered = True
                    st.warning(texts[language]['storm_alert'])
                if not alerts_triggered:
                    st.success("✅ **Weather is favorable for farming operations.**")
            else:
                st.error("❌ Unable to fetch weather data. Check API key, internet connection, or coordinates.")

with tab3:
    st.subheader(f"{texts[language]['tab3']}")
    
   
    historical_data = generate_historical_data(selected_crop)
    
    # --- PREMIUM GREENGLASS HISTORICAL CHARTS ---
    # --- ULTRA PROFESSIONAL SMOOTH HISTORICAL CHARTS ---

    def smooth_line(color):
        return dict(
            color=color,
            width=5,
            shape="spline",          # <-- MAGIC: makes line smooth
            smoothing=1.3,           # <-- controls curve softness
        )

    fig_history = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>📈 Yield Trend</b>",
            "<b>🌧️🌡️ Climate Data</b>",
            "<b>🌱 Soil Parameters</b>",
            "<b>💰 Economic Indicators</b>"
        )
    )

    # ------------------- YIELD TREND -------------------
    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Yield'],
            name='Yield',
            mode="lines+markers",
            line=smooth_line("#00c853"),
            marker=dict(size=8, color="#00c853"),
            hovertemplate="Yield: %{y} tons<br>%{x}"
        ),
        row=1, col=1
    )

    # ------------------- CLIMATE DATA -------------------
    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Rainfall'],
            name='Rainfall',
            mode="lines+markers",
            line=smooth_line("#2962ff"),
            marker=dict(size=8, color="#2962ff"),
            hovertemplate="Rainfall: %{y} mm<br>%{x}"
        ),
        row=1, col=2
    )

    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Temperature'],
            name='Temperature',
            mode="lines+markers",
            line=smooth_line("#ff1744"),
            marker=dict(size=8, color="#ff1744"),
            hovertemplate="Temp: %{y}°C<br>%{x}"
        ),
        row=1, col=2
    )

    # ------------------- SOIL PARAMETERS -------------------
    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Soil_Moisture'],
            name='Soil Moisture',
            mode="lines+markers",
            line=smooth_line("#6d4c41"),
            marker=dict(size=8, color="#6d4c41"),
            hovertemplate="Moisture: %{y}%<br>%{x}"
        ),
        row=2, col=1
    )

    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Soil_pH'],
            name='Soil pH',
            mode="lines+markers",
            line=smooth_line("#aa00ff"),
            marker=dict(size=8, color="#aa00ff"),
            hovertemplate="pH: %{y}<br>%{x}"
        ),
        row=2, col=1
    )

    # ------------------- ECONOMIC INDICATORS -------------------
    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Market_Price'],
            name='Market Price',
            mode="lines+markers",
            line=smooth_line("#ffab00"),
            marker=dict(size=8, color="#ffab00"),
            hovertemplate="Market Price: ₹%{y}<br>%{x}"
        ),
        row=2, col=2
    )

    fig_history.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Labor_Cost'],
            name='Labor Cost',
            mode="lines+markers",
            line=smooth_line("#424242"),
            marker=dict(size=8, color="#424242"),
            hovertemplate="Labor Cost: ₹%{y}<br>%{x}"
        ),
        row=2, col=2
    )

    # ------------------- PREMIUM LAYOUT -------------------
    fig_history.update_layout(
        height=850,
        plot_bgcolor="rgba(255,255,255,0.45)",
        paper_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        )
    )

    # --- SMOOTH GRID LINES ---
    fig_history.update_xaxes(
        showgrid=True,
        gridwidth=0.4,
        gridcolor="rgba(0,0,0,0.1)",
        tickangle=45
    )
    fig_history.update_yaxes(
        showgrid=True,
        gridwidth=0.4,
        gridcolor="rgba(0,0,0,0.1)"
    )

    st.plotly_chart(fig_history, use_container_width=True)


    
    st.markdown(texts[language]["title1"])
    st.markdown("""
    - {}: {:.2f} tons/ha  
    - {}: {:.2f} mm  
    - {}: {:.2f}°C  
    - {}: {:.2f} kg/ha  
    - {}: {:.2f}%  
    - {}: {:.2f}  
    - {}: ₹{:.2f}/ton  
    - {}: ₹{:.2f}/ha
    """.format(
        texts[language]["yield"], historical_data['Yield'].mean(),
        texts[language]["rainfall"], historical_data['Rainfall'].mean(),
        texts[language]["temperature"], historical_data['Temperature'].mean(),
        texts[language]["fertilizer"], historical_data['Fertilizer_Used'].mean(),
        texts[language]["moisture"], historical_data['Soil_Moisture'].mean(),
        texts[language]["ph"], historical_data['Soil_pH'].mean(),
        texts[language]["price"], historical_data['Market_Price'].mean(),
        texts[language]["labor"], historical_data['Labor_Cost'].mean()
    ))

with tab4:
    data = farmer_benefits[language]

    st.header(data["title"])
    st.markdown(data["header"])

    for point in data["points"]:
        st.markdown(f"- {point}")

    st.markdown("---")
    st.markdown(data["summary_title"])

    for item in data["summary"]:
        st.markdown(f"- {item}")

    st.success(data["footer"])