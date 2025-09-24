# 🌏 GreenGrow AI - Smart Soil Analysis System  

GreenGrow AI is an **AI-powered soil analysis and fertilizer recommendation system** built with **Streamlit**.  
It integrates **real-time weather data, soil parameter monitoring, fertilizer recommendations, and historical trend analysis** to help farmers make data-driven decisions.  

---

## ✨ Features  

### 🔍 Soil Analysis  
- Enter or simulate soil parameters (pH, N, P, K, moisture, micronutrients, etc.)  
- Get instant soil health reports and gauge visualizations  
- Fertilizer recommendations based on crop-specific nutrient needs  

### 🌤️ Weather Forecast Integration  
- Fetch **real-time weather data** using the [OpenWeatherMap API](https://openweathermap.org/api)  
- View temperature, rainfall, humidity, wind speed, and more  
- Smart **farming alerts** for heat stress, drought, rainfall, storms, and humidity  

### 📈 Historical Data Insights  
- Generate simulated past yield, rainfall, temperature, soil moisture, and economic indicators  
- Visualize yield trends, climate patterns, soil health, and market data  

### 🌱 Crop Database  
- Includes nutrient requirements, growth stages, varieties, yield potential, and seasonality  
- Supports major crops like **rice, wheat, maize, soybean, cotton, potato, tomato, sugarcane, millets, pulses, oilseeds, and vegetables**  

---

## 🛠️ Installation  

1. **Clone the repository**  
```bash
   git clone https://github.com/Rahul-pr-0503/Fertilizer_Recommendation.git
   cd greengrow-ai
   Create a virtual environment (optional but recommended)

    bash
    Copy code
    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
    Install dependencies

    bash
    Copy code
    pip install -r requirements.txt
    Run the app

    bash
    Copy code
    streamlit run app.py
```

### 🔑 API Key Setup
- This app requires an OpenWeatherMap API key:
- Go to OpenWeatherMap
- Sign up for a free account
- Get your API key under "My API Keys"
- Enter it in the sidebar when running the app

## 📊 Tech Stack
- Frontend/UI: Streamlit
- Data Visualization: Plotly
- ML Model Loading: Joblib
- Weather Data: OpenWeatherMap API
- Data Processing: Pandas, NumPy