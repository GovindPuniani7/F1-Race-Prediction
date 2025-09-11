## F1 Race Position Predictor 

An end-to-end ML project that predicts a Formula 1 driver's final race position using qualifying results and contextual features (track type, team, year, nationality). It includes a Streamlit app with telemetry comparison, explainable predictions (SHAP), prediction history, and track advantage insights.

### Features
- End-to-end pipeline: data merge, feature engineering, train/test split, XGBoost model
- SHAP summary during training, plus per-prediction explanation in the app
- Telemetry mode powered by FastF1 to compare drivers on any session lap
- Persistent prediction history with error tracking
- Track advantage scores (average positions gained per track) and visualization

### Tech Stack
- Data: FastF1, Kaggle-derived F1 datasets (CSV)
- Processing: Pandas, NumPy
- Modeling: Scikit-learn, XGBoost
- Explainability: SHAP
- Viz: Matplotlib, Seaborn, Streamlit

### Project Structure
- `f1_project.py`: trains model, saves `model_xgb.pkl`, `model_features.csv`, `shap_summary_plot.png`, `track_advantage_scores.csv`
- `app.py`: Streamlit app with Quick Prediction, Insights, Telemetry modes
- `data/`: CSVs (races, results, qualifying, drivers, constructors, etc.)
- `f1_cache/`: FastF1 cache for fast repeat telemetry loads

### Setup
1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Train the model and export artifacts
```bash
python f1_project.py
```
4) Run the Streamlit app
```bash
streamlit run app.py
```

### Using the App
- Quick Prediction: choose driver/team/track/year/quali position → get predicted finish + SHAP factors
- Insights: view and edit prediction history, see average error, visualize track advantage
- Telemetry: load any session, compare drivers on speed/throttle/brake/RPM/gear with delta speed plot

### Notes
- Feature alignment: app uses `tracktype_street` to match training one-hot naming
- Caching: FastF1 uses `f1_cache/` directory; first-time loads may take longer

### Resume Highlights
- Built an end-to-end ML pipeline to predict F1 race finishing positions (RMSE printed after training)
- Engineered categorical and contextual features; used XGBoost as the final model
- Implemented explainable AI with SHAP: global summary and per-prediction explanations
- Designed a Streamlit dashboard: telemetry comparison, interactive predictions, and insights
- Persisted prediction history and created track advantage score analytics


