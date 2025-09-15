import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import json
import time

# Lazy-load heavy libraries to improve startup time and prevent errors
try:
    import shap
except ImportError:
    shap = None
try:
    import fastf1
    from fastf1 import plotting as ff_plotting
except ImportError:
    fastf1 = None

# ---------------- CONFIG & ASSET LOADING ----------------
st.set_page_config(page_title="F1 Intelligence Hub", page_icon="🏎️", layout="wide")

@st.cache_resource
def load_assets():
    """Load all critical model artifacts."""
    model_path = "model_xgb.pkl"
    features_path = "model_features.csv"
    metrics_path = "metrics.json"
    test_predictions_path = "test_predictions_vs_actual.csv"

    for f_path in [model_path, features_path, metrics_path, test_predictions_path]:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Critical file not found: {f_path}. Please run the training script and ensure all artifacts are in the repository.")
    
    model = joblib.load(model_path)
    model_features_list = pd.read_csv(features_path)['feature'].tolist()
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    test_predictions = pd.read_csv(test_predictions_path)
    return model, model_features_list, metrics, test_predictions

try:
    model, model_features_list, metrics, test_predictions = load_assets()
    explainer = shap.TreeExplainer(model) if model and shap else None
except Exception as e:
    st.error(f"## 💥 App failed to start! 💥")
    st.error(f"**Error:** Could not load essential files. Please ensure `model_xgb.pkl`, `model_features.csv`, `metrics.json`, and `test_predictions_vs_actual.csv` are in your GitHub repository.")
    st.stop()

# ---------------- HELPERS & UI ----------------
TEAM_COLORS = {"Mercedes": "#00D2BE","Red Bull": "#1E41FF","Ferrari": "#DC0000","Alpine": "#FD5DA8","McLaren": "#FF8700","Aston Martin": "#006F62","Williams": "#005AFF","Haas": "#B6BABD","AlphaTauri": "#2B4562","Alfa Romeo": "#981E32"}
if "logs" not in st.session_state: st.session_state["logs"] = []

def create_prediction_features(inputs: dict, model_features_list: list, metrics: dict) -> pd.DataFrame:
    """Builds a 1-row DataFrame aligned to the model's feature list."""
    base_features = {
        "position_qual": inputs.get("position_qual", 10),
        "year": inputs.get("year", 2023),
        "tracktype_street": 1 if inputs.get("track_type", "circuit") == "street" else 0,
        "driver_form_5": metrics.get("driver_form_median", 10.0),
        "team_form_5": metrics.get("team_form_median", 10.0)
    }
    df = pd.DataFrame([base_features])
    for prefix, value in [("driver", inputs.get("driver")), ("team", inputs.get("team")), ("track", inputs.get("track")), ("nat", inputs.get("nationality"))]:
        if value:
            df[f"{prefix}_{value}"] = 1
    final_df = df.reindex(columns=model_features_list, fill_value=0)
    return final_df

def get_feature_explanation(feature_name, shap_value, selected_team, selected_year):
    """Generates a plain-English explanation for a given feature's impact."""
    explanation, feature_name_lower = "", feature_name.lower()
    if 'year' in feature_name_lower: explanation = f"_(This may reflect the {selected_team} car's competitiveness in {selected_year}.)_"
    elif 'team' in feature_name_lower: explanation = f"_(The model has learned the general performance level of the {selected_team} team.)_"
    elif 'qual' in feature_name_lower: explanation = "_(A high grid position is key.)_" if shap_value < 0 else "_(Starting further down is a major disadvantage.)_"
    return explanation

def analyze_telemetry(lap_d1, lap_d2):
    """Generates a markdown string with key insights from telemetry data."""
    lap_time_diff = lap_d1['LapTime'] - lap_d2['LapTime']
    faster_driver = lap_d1['Driver'] if lap_time_diff.total_seconds() < 0 else lap_d2['Driver']
    summary = f"#### Lap Analysis: {lap_d1['Driver']} vs {lap_d2['Driver']}\n"
    summary += f"- **Overall Lap Time:** **{faster_driver}** was faster by **{abs(lap_time_diff.total_seconds()):.3f}** seconds.\n"
    tel_d1 = lap_d1.get_car_data(); tel_d2 = lap_d2.get_car_data()
    top_speed_d1 = tel_d1['Speed'].max(); top_speed_d2 = tel_d2['Speed'].max()
    faster_top_speed_driver = lap_d1['Driver'] if top_speed_d1 > top_speed_d2 else lap_d2['Driver']
    summary += f"- **Top Speed:** **{faster_top_speed_driver}** reached a higher top speed of **{max(top_speed_d1, top_speed_d2):.0f} km/h**.\n"
    return summary

with st.sidebar:
    st.header("F1 Intelligence Hub 🏎️")
    team = st.selectbox("Select Your Team:", list(TEAM_COLORS.keys()), key="team")
    theme_color = TEAM_COLORS.get(team, "#FFFFFF")
    st.divider()
    st.header("Navigation")
    page = st.radio("Choose Your Mission:", ["🏠 Home", "🎯 Quick Prediction", "📈 Model Performance", "📊 Insights & History", "📦 Batch Prediction", "📡 Telemetry"], label_visibility="collapsed")

st.markdown(f"<style>@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');html,body,[class*='st-']{{font-family:'Roboto',sans-serif;}}.stApp{{background-color:#0E1117;}}h1{{color:#FFFFFF;font-weight:700;border-bottom:2px solid {theme_color};padding-bottom:10px;}}h2,h3{{color:#FAFAFA;}}.card{{background:rgba(40,40,40,0.5);border-radius:15px;padding:25px;margin-bottom:20px;border:1px solid rgba(255,255,255,0.1);}}.metric-card{{text-align:center;}}.metric-card h3{{font-size:18px;color:#A0A0A0;margin-bottom:5px;}}.metric-card p{{font-size:32px;font-weight:700;color:{theme_color};}}.prediction-card .value{{font-size:48px;font-weight:700;}}.positive{{color:#00e676;}}.neutral{{color:#ffeb3b;}}.negative{{color:#ff1744;}}</style>", unsafe_allow_html=True)

# ---------------- PAGE ROUTING ----------------
if page == "🏠 Home":
    st.title("F1 Intelligence Hub")
    st.image("https://media.formula1.com/image/upload/f_auto,c_limit,w_1920,q_auto/f_auto/q_auto/fom-website/2020/banners/2023/F1%20header%202023", use_container_width=True)
    st.markdown("<div class='card'><p>Welcome to your central dashboard for predicting Formula 1 race outcomes. This tool leverages machine learning to forecast finishing positions and provides deep dives into telemetry data. Use the sidebar to navigate through the app's features.</p></div>", unsafe_allow_html=True)
    if os.path.exists("shap_summary_plot.png"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Global Feature Importance (SHAP)")
        st.image("shap_summary_plot.png", caption="This plot shows the overall impact of features on the model's predictions. Longer bars are more important.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🎯 Quick Prediction":
    st.title("Quick Race Prediction")
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            driver = st.selectbox("Driver", ["hamilton", "verstappen", "leclerc", "norris", "sainz", "perez", "alonso", "russell", "gasly", "ocon", "bottas", "stroll", "tsunoda", "albon", "zhou", "hulkenberg", "magnussen", "piastri"])
            position_qual = st.slider("Qualifying Position", 1, 20, 10)
        with col2:
            track = st.selectbox("Track", ["British Grand Prix", "Monaco Grand Prix", "Abu Dhabi GP", "Australian Grand Prix", "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Italian Grand Prix"])
            year = st.slider("Season Year", 2018, 2024, 2023)
        nationality = st.selectbox("Driver Nationality", ["British", "Dutch", "Monegasque", "Australian", "Spanish", "Mexican", "Finnish", "French", "German", "Canadian", "Japanese"])
        track_type = st.selectbox("Track Type", ["circuit", "street"])
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🏁 Predict Race Position"):
        with st.spinner("🔮 Performing AI magic..."):
            user_inputs = {"driver": driver, "position_qual": position_qual, "year": year, "track": track, "nationality": nationality, "track_type": track_type, "team": team}
            try:
                features = create_prediction_features(user_inputs, model_features_list, metrics)
                prediction = model.predict(features)[0]
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
            else:
                style_class = "positive" if prediction <= 3 else "neutral" if prediction <= 10 else "negative"
                st.markdown(f"<div class='card prediction-card'><p>Predicted Final Position:</p><p class='value {style_class}'>{prediction:.0f}</p></div>", unsafe_allow_html=True)
                log_row = {"Driver": driver, "Team": team, "Track": track, "Quali Pos": position_qual, "Year": year, "Predicted": round(prediction, 2)}
                st.session_state["logs"].append(log_row)

                st.subheader("🧠 AI Reasoning Explained")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                if explainer:
                    try:
                        shap_values = explainer(features)
                        fig, ax = plt.subplots(); shap.plots.bar(shap_values[0], max_display=7, show=False); st.pyplot(fig); plt.close(fig)
                        st.divider()
                        st.markdown("#### Key Factors Breakdown:")
                        shap_list = sorted(list(zip(features.columns, shap_values.values[0])), key=lambda x: abs(x[1]), reverse=True)
                        summary = f"The model predicts a finish of **P{prediction:.0f}**. The most critical factors were:\n"
                        for feature, shap_val in shap_list[:4]:
                            if abs(shap_val) < 0.05: continue
                            clean_feature = feature.replace("_", " ").replace("driverRef", "Driver being").replace("name team", "Team being").replace("position qual", "Qualifying Position of")
                            impact_text = f"**helped** by about {abs(shap_val):.2f} positions" if shap_val < 0 else f"**hurt** by about {abs(shap_val):.2f} positions"
                            reason = get_feature_explanation(feature, shap_val, team, year)
                            summary += f"- The **{clean_feature.title()}** {impact_text}. {reason}\n"
                        st.markdown(summary)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation. Error: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Validation")
    st.markdown("<div class='card'><p>This page validates the model's accuracy by comparing its predictions against the actual race results from a held-out test dataset. A perfect model would have all points lying on the red diagonal line.</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: st.markdown("<div class='card metric-card'><h3>Test Set RMSE</h3><p>{:.2f}</p></div>".format(metrics.get('rmse', 0)), unsafe_allow_html=True)
    with col2: st.markdown("<div class='card metric-card'><h3>Test Set MAE</h3><p>{:.2f}</p></div>".format(metrics.get('mae', 0)), unsafe_allow_html=True)
        
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Predicted vs. Actual Finishing Position")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(test_predictions['Actual_Position'], test_predictions['Predicted_Position'], alpha=0.3, color=theme_color)
    ax.plot([1, 22], [1, 22], 'r--', label='Perfect Prediction')
    ax.set_xlabel("Actual Finishing Position", fontsize=12); ax.set_ylabel("Predicted Finishing Position", fontsize=12)
    ax.set_title("Model Accuracy on Test Set", fontsize=14); ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
    st.pyplot(fig); plt.close(fig)
    
    st.divider()
    st.markdown("#### Interpreting the Results")
    st.markdown(f"""
    - **RMSE (Root Mean Squared Error): {metrics.get('rmse', 0):.2f}**
      - This means that, on average, the model's predictions are off by about {metrics.get('rmse', 0):.2f} positions. It's a standard metric, but sensitive to large errors.
    - **MAE (Mean Absolute Error): {metrics.get('mae', 0):.2f}**
      - This is easier to interpret: on average, the model's prediction is about **{metrics.get('mae', 0):.0f} positions** away from the actual result.
    - **The Scatter Plot:** Each dot represents one race result from the test set. The closer the dots are to the red 'Perfect Prediction' line, the more accurate the model is.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Insights & History":
    st.title("📊 Prediction History")
    hist_path = "predictions_history.csv"
    session_logs_df = pd.DataFrame(st.session_state["logs"])
    if os.path.exists(hist_path):
        file_history_df = pd.read_csv(hist_path)
        combined_logs = pd.concat([file_history_df, session_logs_df]).drop_duplicates(subset=["Driver", "Track", "Year"], keep='last', inplace=False)
    else:
        combined_logs = session_logs_df

    if st.button("💾 Save Session History"):
        if not combined_logs.empty: combined_logs.to_csv(hist_path, index=False); st.success("History saved!")
        else: st.warning("No history to save.")
        
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if combined_logs.empty: st.info("No predictions made yet.")
    else: st.dataframe(combined_logs)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📦 Batch Prediction":
    st.title("📦 Batch Prediction")
    st.markdown("<div class='card'><p>Upload a CSV with scenarios to get predictions. The app will align columns to the model's features.</p></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                input_df = pd.read_csv(uploaded_file)
                aligned_df = input_df.reindex(columns=model_features_list, fill_value=0)
                predictions = model.predict(aligned_df)
                results_df = input_df.copy()
                results_df["predicted_position"] = [round(p, 2) for p in predictions]
                st.markdown("<div class='card'><h3>Batch Predictions</h3></div>", unsafe_allow_html=True)
                st.dataframe(results_df)
            except Exception as e:
                st.error(f"Failed to process file. Error: {e}")

elif page == "📡 Telemetry":
    st.title("📡 Telemetry Battle Mode")
    st.markdown("<div class='card'><p>Compare the fastest laps of any two drivers from a race weekend. This feature is loaded on demand.</p></div>", unsafe_allow_html=True)
    if fastf1 is None:
        st.error("FastF1 library is not available.")
    else:
        year = st.selectbox("Year", [2024, 2023, 2022], key="telemetry_year")
        try:
            events = fastf1.events.get_event_schedule(year, include_testing=False)
            gp = st.selectbox("Grand Prix", events['EventName'].tolist(), key="telemetry_gp")
            session_type = st.selectbox("Session", ["Race", "Qualifying"], key="telemetry_session")

            if st.button("Load & Compare Fastest Laps"):
                with st.spinner(f"Loading session data for {gp}... This may take a moment..."):
                    session = fastf1.get_session(year, gp, session_type)
                    session.load(telemetry=True, weather=False, messages=False)
                    drivers = pd.unique(session.laps['Driver']).tolist()
                    if len(drivers) < 2:
                        st.warning("Not enough driver data for comparison.")
                    else:
                        d1, d2 = drivers[0], drivers[1]
                        
                        lap_d1 = session.laps.pick_driver(d1).pick_fastest()
                        lap_d2 = session.laps.pick_driver(d2).pick_fastest()
                        
                        if pd.isna(lap_d1['LapTime']) or pd.isna(lap_d2['LapTime']):
                            st.error("One driver did not set a valid fastest lap.")
                        else:
                            tel_d1 = lap_d1.get_car_data().add_distance()
                            tel_d2 = lap_d2.get_car_data().add_distance()
                            
                            common_distance = tel_d1['Distance']
                            speed_d2_interp = np.interp(common_distance, tel_d2['Distance'], tel_d2['Speed'])
                            throttle_d2_interp = np.interp(common_distance, tel_d2['Distance'], tel_d2['Throttle'])
                            
                            color_d1 = f"#{session.results.loc[lap_d1['DriverNumber']]['TeamColor']}"
                            color_d2 = f"#{session.results.loc[lap_d2['DriverNumber']]['TeamColor']}"

                            fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
                            ax[0].plot(common_distance, tel_d1['Speed'], color=color_d1, label=d1)
                            ax[0].plot(common_distance, speed_d2_interp, color=color_d2, label=d2)
                            ax[0].set_ylabel('Speed (Km/h)'); ax[0].legend()
                            
                            ax[1].plot(common_distance, tel_d1['Throttle'], color=color_d1, label=d1)
                            ax[1].plot(common_distance, throttle_d2_interp, color=color_d2, label=d2)
                            ax[1].set_ylabel('Throttle (%)')
                            
                            delta_time, ref_tel, _ = fastf1.utils.delta_time(lap_d1, lap_d2)
                            ax[2].plot(ref_tel['Distance'], delta_time, color='white')
                            ax[2].axhline(0, color='grey', linestyle='--')
                            ax[2].set_ylabel(f"Time Delta ({d2} to {d1})")
                            st.pyplot(fig); plt.close(fig)
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown(analyze_telemetry(lap_d1, lap_d2))
                            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load Telemetry data. Error: {e}")
