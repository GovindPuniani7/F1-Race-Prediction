import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import json
import joblib

def load_data(data_path="data/"):
    """Loads all necessary CSV files from the data directory."""
    print("🔄 Loading CSV files...")
    races = pd.read_csv(data_path + "races.csv")
    results = pd.read_csv(data_path + "results.csv")
    qualifying = pd.read_csv(data_path + "qualifying.csv")
    drivers = pd.read_csv(data_path + "drivers.csv")
    constructors = pd.read_csv(data_path + "constructors.csv")
    return races, results, qualifying, drivers, constructors

def engineer_features(races, results, qualifying, drivers, constructors):
    """Merges, cleans, and engineers features for the F1 model."""
    print("⚙️ Engineering features...")
    df = results.merge(races[['raceId', 'year', 'name', 'date']], on='raceId', how='left')
    df = df.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left', suffixes=('', '_qual'))
    df = df.merge(drivers[['driverId', 'driverRef', 'nationality']], on='driverId', how='left')
    df = df.merge(constructors[['constructorId', 'name']], on='constructorId', how='left', suffixes=('', '_team'))

    df.dropna(subset=['position_qual', 'position'], inplace=True)
    df = df[df['position'] != '\\N']
    df['position_qual'] = df['position_qual'].astype(int)
    df['position'] = df['position'].astype(int)
    
    street_circuits = ['Monaco Grand Prix', 'Singapore Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Las Vegas Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix']
    df['track_type'] = df['name'].apply(lambda x: 'street' if x in street_circuits else 'circuit')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['driverId', 'date'])
    df['driver_form_5'] = df.groupby('driverId')['position'].shift(1).rolling(window=5, min_periods=1).mean()
    df = df.sort_values(['constructorId', 'date'])
    df['team_form_5'] = df.groupby('constructorId')['position'].shift(1).rolling(window=5, min_periods=1).mean()
    
    driver_form_median = df['driver_form_5'].median()
    team_form_median = df['team_form_5'].median()

    df.fillna({'driver_form_5': driver_form_median, 'team_form_5': team_form_median}, inplace=True)
    return df, driver_form_median, team_form_median

def create_feature_matrix(df):
    """Creates the final feature matrix (X) and target vector (y)."""
    print(" MATRIX CREATION ".center(40, '-'))
    X = pd.concat([
        df[['position_qual', 'year', 'driver_form_5', 'team_form_5']],
        pd.get_dummies(df['driverRef'], prefix='driver'),
        pd.get_dummies(df['name_team'], prefix='team'),
        pd.get_dummies(df['name'], prefix='track'),
        pd.get_dummies(df['nationality'], prefix='nat'),
        pd.get_dummies(df['track_type'], prefix='tracktype')
    ], axis=1)
    y = df['position']
    return X, y

def train_model(X, y):
    """Trains the XGBoost model and returns the model and test data."""
    print("🧠 Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"✅ Model trained! 🎯 Final Test RMSE: {final_rmse:.2f}")
    print(f"✅ Final Test MAE: {final_mae:.2f}")
    
    return model, X, y_test, y_pred, final_rmse, final_mae

def save_artifacts(model, X, y_test, y_pred, final_rmse, final_mae, driver_form_median, team_form_median):
    """Saves all necessary model artifacts."""
    print("💾 Saving artifacts...")
    joblib.dump(model, "model_xgb.pkl")

    model_features = pd.DataFrame(X.columns, columns=['feature'])
    model_features.to_csv("model_features.csv", index=False)

    metrics = {
        "rmse": float(final_rmse),
        "mae": float(final_mae),
        "features": int(X.shape[1]),
        "driver_form_median": float(driver_form_median),
        "team_form_median": float(team_form_median)
    }
    with open("metrics.json", "w") as f: json.dump(metrics, f, indent=4)
        
    # NEW: Save test predictions for the app to visualize
    test_results = pd.DataFrame({'Actual_Position': y_test, 'Predicted_Position': y_pred})
    test_results.to_csv('test_predictions_vs_actual.csv', index=False)
    
    print("✅ Test predictions saved to test_predictions_vs_actual.csv")

    # Generate and save SHAP summary plot
    explainer = shap.Explainer(model)
    shap_values = explainer(X.iloc[:200]) # Use a sample for speed
    plt.figure()
    shap.summary_plot(shap_values, X.iloc[:200], show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight')
    plt.close()

    print("✅ All artifacts saved successfully.")

def main():
    """Main function to run the entire ML pipeline."""
    data = load_data()
    if data:
        races, results, qualifying, drivers, constructors = data
        engineered_df, driver_form_median, team_form_median = engineer_features(races, results, qualifying, drivers, constructors)
        X, y = create_feature_matrix(engineered_df)
        model, X, y_test, y_pred, final_rmse, final_mae = train_model(X, y)
        save_artifacts(model, X, y_test, y_pred, final_rmse, final_mae, driver_form_median, team_form_median)
        print("\n🎉 F1 Project pipeline finished successfully! 🎉")

if __name__ == "__main__":
    main()