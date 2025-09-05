import streamlit as st
import pandas as pd
import joblib
import fastf1
from fastf1 import plotting
import matplotlib.pyplot as plt
import os
import shap
import json
import numpy as np

# ---------------- CONFIG & CACHE ----------------
st.set_page_config(page_title="F1 Intelligence Hub", page_icon="🏎️", layout="wide")

# --- FastF1 Cache Setup ---
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)
plotting.setup_mpl()

# ---------------- LOAD MODEL & ASSETS ----------------
@st.cache_resource
def load_assets():
    """Load all cached assets like model and features."""
    try:
        model = joblib.load("model_xgb.pkl")
        model_features = pd.read_csv("model_features.csv")
        return model, model_features
    except FileNotFoundError:
        st.error("Critical files not found. Please ensure 'model_xgb.pkl' and 'model_features.csv' are present.")
        return None, None

model, model_features = load_assets()
if model:
    explainer = shap.TreeExplainer(model)

# ---------------- TEAM COLORS & SESSION STATE ----------------
TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Red Bull": "#1E41FF", "Ferrari": "#DC0000",
    "Alpine": "#FD5DA8", "McLaren": "#FF8700", "Aston Martin": "#006F62",
    "Williams": "#005AFF", "Haas": "#B6BABD", "AlphaTauri": "#2B4562",
    "Alfa Romeo": "#981E32"
}
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# ---------------- SIDEBAR SETUP ----------------
with st.sidebar:
    st.header("F1 Intelligence Hub 🏎️")
    team = st.selectbox("Select Your Team:", list(TEAM_COLORS.keys()), key="team")
    theme_color = TEAM_COLORS[team]
    st.divider()
    st.header("Navigation")
    page = st.radio("Choose Your Mission:",
                    ["🏠 Home", "🎯 Quick Prediction", "📊 Insights & History", "📦 Batch Prediction", "📡 Telemetry"],
                    label_visibility="collapsed")

# ---------------- DESIGNER CSS STYLES ----------------
st.markdown(f"""
    <style>
        /* CSS styles remain the same as the previous version */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="st-"] {{ font-family: 'Roboto', sans-serif; }}
        .stApp {{ background-color: #0E1117; }}
        h1 {{ color: #FFFFFF; font-weight: 700; border-bottom: 2px solid {theme_color}; padding-bottom: 10px; }}
        h2, h3 {{ color: #FAFAFA; }}
        .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {{ color: {theme_color}; }}
        .card {{ background: rgba(40, 40, 40, 0.5); border-radius: 15px; padding: 25px; margin-bottom: 20px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }}
        .metric-card {{ text-align: center; }}
        .metric-card h3 {{ font-size: 18px; color: #A0A0A0; margin-bottom: 5px; }}
        .metric-card p {{ font-size: 32px; font-weight: 700; color: {theme_color}; }}
        .prediction-card {{ text-align: center; }}
        .prediction-card .value {{ font-size: 48px; font-weight: 700; }}
        .positive {{ color: #00e676; }}
        .neutral {{ color: #ffeb3b; }}
        .negative {{ color: #ff1744; }}
        .stButton>button {{ border: 2px solid {theme_color}; border-radius: 25px; color: {theme_color}; padding: 10px 25px; background-color: transparent; font-weight: bold; transition: all 0.3s ease; }}
        .stButton>button:hover {{ background-color: {theme_color}; color: #0E1117; border-color: {theme_color}; }}
    </style>
""", unsafe_allow_html=True)


# ---------------- PAGE ROUTING ----------------

if page == "🏠 Home":
    # This page logic is unchanged
    st.title("F1 Intelligence Hub")
    st.image("https://media.formula1.com/image/upload/f_auto,c_limit,w_1920,q_auto/f_auto/q_auto/fom-website/2020/banners/2023/F1%20header%202023", use_container_width=True)
    st.markdown("<div class='card'><p>Welcome to your central dashboard for predicting Formula 1 race outcomes. This tool leverages machine learning to forecast finishing positions and provides deep dives into telemetry data. Use the sidebar to select your favorite team and navigate through the app's features.</p></div>", unsafe_allow_html=True)
    st.subheader("Live Model Metrics")
    try:
        with open("metrics.json", "r", encoding="utf-8") as f: m = json.load(f)
        rmse, cv_rmse, features = f"{m.get('rmse', 'N/A'):.2f}", f"{m.get('cv_rmse', 'N/A'):.2f}", f"{m.get('features', 'N/A')}"
    except Exception:
        rmse, cv_rmse, features = "N/A", "N/A", "N/A"
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"<div class='card metric-card'><h3>Model RMSE</h3><p>{rmse}</p></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='card metric-card'><h3>CV RMSE (5-fold)</h3><p>{cv_rmse}</p></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='card metric-card'><h3>Features</h3><p>{features}</p></div>", unsafe_allow_html=True)


elif page == "🎯 Quick Prediction":
    st.title("Quick Race Prediction")

    # --- UI FIX: MOVED INPUTS TO MAIN PAGE IN COLUMNS ---
    st.subheader("Scenario Configuration")
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            driver = st.selectbox("Driver", ["hamilton", "verstappen", "leclerc", "norris", "sainz", "perez", "alonso", "russell", "gasly", "ocon", "bottas", "stroll", "tsunoda", "albon", "zhou", "hulkenberg", "magnussen", "piastri"])
            position_qual = st.slider("Qualifying Position", 1, 20, 10)
            year = st.slider("Season Year", 2018, 2024, 2023)
        with col2:
            track = st.selectbox("Track", ["British Grand Prix", "Monaco Grand Prix", "Abu Dhabi GP", "Australian Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Italian Grand Prix"])
            nationality = st.selectbox("Driver Nationality", ["British", "Dutch", "Monegasque", "Australian", "Spanish", "Mexican", "Finnish", "French", "German", "Canadian", "Japanese"])
            track_type = st.selectbox("Track Type", ["circuit", "street"])
        st.markdown("</div>", unsafe_allow_html=True)

    if not model:
        st.warning("Model is not loaded. Cannot perform predictions.")
    else:
        if st.button("🏁 Predict Race Position"):
            with st.spinner("🔮 Performing AI magic..."):
                # --- Feature Engineering Logic (Unchanged) ---
                features = pd.DataFrame({"position_qual": [position_qual], "year": [year], "tracktype_street": [1 if track_type == "street" else 0]})
                for col, val, prefix in [("driverRef", driver, "driver"), ("name", track, "track"), ("name_team", team, "team"), ("nationality", nationality, "nat")]:
                    encoded = pd.get_dummies(pd.Series([val]), prefix=prefix)
                    features = features.join(encoded)
                for col in model_features.columns:
                    if col not in features.columns: features[col] = 0
                features = features[model_features.columns]

                prediction = model.predict(features)[0]
                style_class = "positive" if prediction <= 3 else "neutral" if prediction <= 10 else "negative"

                st.markdown(f"<div class='card prediction-card'><p>Predicted Final Position:</p><p class='value {style_class}'>{prediction:.0f}</p></div>", unsafe_allow_html=True)
                
                # --- UX UPGRADE: WRITTEN EXPLANATION OF AI REASONING ---
                st.subheader("🧠 AI Reasoning Explained")
                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    shap_values = explainer(features)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.bar(shap_values[0], max_display=7, show=False)
                    plt.xlabel("Impact on Predicted Finishing Position")
                    st.pyplot(fig, use_container_width=True)
                    st.caption("💡 Bars pushing left (blue) help the driver finish higher (a lower position number). Bars pushing right (red) hurt the finish position.")

                    st.divider()
                    st.markdown("#### Key Factors Breakdown:")
                    
                    # --- NEW: Helper function for generating "why" explanations ---
                    def get_feature_explanation(feature_name, shap_value, selected_team, selected_year):
                        explanation = ""
                        # Explanation for Year
                        if 'year' in feature_name:
                            if shap_value > 0: # Negative impact
                                explanation = f"_(This might be because the {selected_team} car was less competitive in {selected_year} compared to other seasons in the data.)_"
                            else: # Positive impact
                                explanation = f"_(This likely reflects that the {selected_team} car was particularly strong during the {selected_year} season.)_"
                        # Explanation for Team
                        elif 'team' in feature_name:
                             if shap_value > 0: # Negative impact
                                explanation = f"_(The model considers {selected_team} to be at a slight disadvantage compared to top-tier teams like Red Bull or Mercedes.)_"
                             else: # Positive impact
                                explanation = f"_(Being a top-tier team, {selected_team} is considered to have a significant advantage by the model.)_"
                        # Explanation for Qualifying
                        elif 'qual' in feature_name:
                            if shap_value > 0: # Negative impact
                                explanation = f"_(Starting further down the grid is a major disadvantage.)_"
                            else:
                                explanation = f"_(A high grid position is a key predictor of a good race result.)_"
                        return explanation


                    shap_list = sorted(list(zip(features.columns, shap_values.values[0])), key=lambda x: abs(x[1]), reverse=True)
                    
                    summary = f"The model predicts a finish of **P{prediction:.0f}**. The most critical factors were:\n"
                    for feature, shap_val in shap_list[:4]:
                        if abs(shap_val) < 0.05: continue

                        clean_feature = feature.replace("_", " ").replace("driverRef", "Driver being").replace("name team", "Team being").replace("position qual", "Qualifying Position of")
                        
                        if shap_val < 0:
                            impact_text = f"**significantly helped** by about {abs(shap_val):.2f} positions"
                        else:
                            impact_text = f"**negatively impacted** by about {abs(shap_val):.2f} positions"
                        
                        # Get the "why" explanation
                        reason = get_feature_explanation(feature, shap_val, team, year)

                        summary += f"- The **{clean_feature.title()}** {impact_text}. {reason}\n"
                    
                    st.markdown(summary)

                    st.markdown("</div>", unsafe_allow_html=True)
elif page == "📊 Insights & History":
    st.title("Insights & Prediction History")
    st.markdown("<div class='card'><p>Review past predictions made during this session and from previous runs. This log helps track model performance over time. You can clear the history at any time.</p></div>", unsafe_allow_html=True)

    hist_path = "predictions_history.csv"
    if st.button("🗑️ Clear Prediction History"):
        if os.path.exists(hist_path):
            os.remove(hist_path)
        st.session_state["logs"] = []
        st.success("Prediction history cleared.")
        st.rerun()

    file_history = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()
    session_history = pd.DataFrame(st.session_state.get("logs", []))
    logs_df = pd.concat([file_history, session_history], ignore_index=True).drop_duplicates(keep='last')

    if not logs_df.empty:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.dataframe(logs_df)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No predictions have been made yet. Go to the 'Quick Prediction' page to start.")


elif page == "📦 Batch Prediction":
    st.title("Batch Scenario Prediction")
    st.markdown("<div class='card'><p>Upload a CSV file with multiple race scenarios to get predictions for all of them at once. Download the template to see the required format.</p></div>", unsafe_allow_html=True)

    template_df = pd.DataFrame({"position_qual": [10, 5], "year": [2023, 2023], "tracktype_street": [0, 1], "driver_hamilton": [1, 0], "driver_leclerc": [0, 1]})
    st.download_button("📥 Download Template CSV", data=template_df.to_csv(index=False), file_name="batch_template.csv")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None and model:
        with st.spinner("Processing batch file..."):
            input_df = pd.read_csv(uploaded_file)
            output_df = pd.DataFrame(columns=model_features.columns)
            output_df = pd.concat([output_df, input_df], ignore_index=True).fillna(0)
            output_df = output_df[model_features.columns]

            predictions = model.predict(output_df)
            results_df = input_df.copy()
            results_df["predicted_position"] = [round(p, 2) for p in predictions]
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Batch Predictions")
            st.dataframe(results_df)
            st.download_button("📥 Download Predictions", data=results_df.to_csv(index=False), file_name="batch_predictions.csv")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "📡 Telemetry":
    st.title("Telemetry Battle Mode")
    st.markdown("<div class='card'><p>Dive deep into the data. Select a session, two drivers, and a lap to compare their telemetry data side-by-side. See who was faster through which corner and how they managed speed, throttle, and braking.</p></div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.selectbox("Year", [2024, 2023, 2022, 2021], key="telemetry_year")
        with col2:
            try:
                events = fastf1.events.get_event_schedule(year, include_testing=False)
                gp = st.selectbox("Grand Prix", events['EventName'].tolist(), key="telemetry_gp")
            except Exception:
                st.warning("Could not load Grand Prix list for this year.")
                gp = None
        with col3:
            session_type = st.selectbox("Session", ["Race", "Qualifying"], key="telemetry_session")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if gp:
        try:
            session = fastf1.get_session(year, gp, session_type)
            session.load(telemetry=False, weather=False, messages=False)
            driver_abbreviations = pd.unique(session.laps['Driver']).tolist()
            
            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                col_d1, col_d2, col_lap = st.columns(3)
                with col_d1:
                    driver1_abbr = st.selectbox("Driver 1", driver_abbreviations, index=0, key="d1")
                with col_d2:
                    driver2_abbr = st.selectbox("Driver 2", driver_abbreviations, index=1, key="d2")
                with col_lap:
                    lap_choice = st.text_input("Lap Number", value="Fastest", key="lap_num")
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Compare Laps"):
                if driver1_abbr == driver2_abbr:
                    st.warning("Please select two different drivers.")
                else:
                    with st.spinner(f"Loading and analyzing data for {year} {gp}..."):
                        session.load()
                        if lap_choice.lower() == 'fastest':
                            lap_d1 = session.laps.pick_driver(driver1_abbr).pick_fastest()
                            lap_d2 = session.laps.pick_driver(driver2_abbr).pick_fastest()
                        else:
                            lap_d1 = session.laps.pick_driver(driver1_abbr).pick_lap(int(lap_choice)).iloc[0]
                            lap_d2 = session.laps.pick_driver(driver2_abbr).pick_lap(int(lap_choice)).iloc[0]

                        tel_d1 = lap_d1.get_car_data().add_distance()
                        tel_d2 = lap_d2.get_car_data().add_distance()
                        
                        color_d1 = fastf1.plotting.team_color(lap_d1['Team'])
                        color_d2 = fastf1.plotting.team_color(lap_d2['Team'])

                        fig, ax = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 2, 1]})
                        # Plotting logic remains the same...
                        ax[0].plot(tel_d1['Distance'], tel_d1['Speed'], color=color_d1, label=driver1_abbr)
                        ax[0].plot(tel_d2['Distance'], tel_d2['Speed'], color=color_d2, label=driver2_abbr)
                        ax[0].set_ylabel('Speed (Km/h)'); ax[0].legend(); ax[0].grid(True, linestyle='--', alpha=0.5)
                        
                        ax[1].plot(tel_d1['Distance'], tel_d1['Throttle'], color=color_d1, label=driver1_abbr)
                        ax[1].plot(tel_d2['Distance'], tel_d2['Throttle'], color=color_d2, label=driver2_abbr)
                        ax[1].set_ylabel('Throttle (%)'); ax[1].grid(True, linestyle='--', alpha=0.5)

                        delta_time, _, _ = fastf1.utils.delta_time(lap_d1, lap_d2)
                        ax[2].plot(tel_d1['Distance'], delta_time, color='white')
                        ax[2].axhline(0, color='grey', linestyle='--')
                        ax[2].set_ylabel(f"Time Delta ({driver2_abbr} to {driver1_abbr})"); ax[2].grid(True, linestyle='--', alpha=0.5)
                        
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not load data for this session. It may not have data available. Error: {e}")