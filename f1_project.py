import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import json
import joblib


def load_data(data_path="data/"):
    """Loads all necessary CSV files from the data directory."""
    print("🔄 Loading CSV files...")
    try:
        races = pd.read_csv(data_path + "races.csv")
        results = pd.read_csv(data_path + "results.csv")
        qualifying = pd.read_csv(data_path + "qualifying.csv")
        drivers = pd.read_csv(data_path + "drivers.csv")
        constructors = pd.read_csv(data_path + "constructors.csv")
        print("✅ CSVs Loaded Successfully.")
        return races, results, qualifying, drivers, constructors
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Make sure all CSV files are in the '{data_path}' directory.")
        return None


def engineer_features(races, results, qualifying, drivers, constructors):
    """Merges, cleans, and engineers features for the F1 model."""
    print("⚙️ Engineering features...")

    df = results.merge(races[['raceId', 'year', 'name', 'date']], on='raceId', how='left')
    df = df.merge(qualifying[['raceId', 'driverId', 'position']], on=['raceId', 'driverId'], how='left',
                  suffixes=('', '_qual'))
    df = df.merge(drivers[['driverId', 'driverRef', 'nationality']], on='driverId', how='left')
    df = df.merge(constructors[['constructorId', 'name']], on='constructorId', how='left', suffixes=('', '_team'))

    df.dropna(subset=['position_qual', 'position'], inplace=True)
    df = df[df['position'] != '\\N']
    df['position_qual'] = df['position_qual'].astype(int)
    df['position'] = df['position'].astype(int)
    df['position_change'] = df['position_qual'] - df['position']

    street_circuits = ['Monaco Grand Prix', 'Singapore Grand Prix', 'Azerbaijan Grand Prix', 'Miami Grand Prix',
                       'Las Vegas Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix']
    df['track_type'] = df['name'].apply(lambda x: 'street' if x in street_circuits else 'circuit')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['driverId', 'date'])
    df['driver_form_5'] = df.groupby('driverId')['position'].shift(1).rolling(window=5, min_periods=1).mean()
    df = df.sort_values(['constructorId', 'date'])
    df['team_form_5'] = df.groupby('constructorId')['position'].shift(1).rolling(window=5, min_periods=1).mean()
    df.fillna({'driver_form_5': df['driver_form_5'].median(), 'team_form_5': df['team_form_5'].median()}, inplace=True)

    print("✅ Feature engineering complete.")
    return df


def create_feature_matrix(df):
    """Creates the final feature matrix (X) and target vector (y)."""
    print(" MATRIX CREATION ".center(40, '-'))
    X = pd.concat([
        df[['position_qual', 'year', 'driver_form_5', 'team_form_5']],
        pd.get_dummies(df['driverRef'], prefix='driver', drop_first=True),
        pd.get_dummies(df['name_team'], prefix='team', drop_first=True),
        pd.get_dummies(df['name'], prefix='track', drop_first=True),
        pd.get_dummies(df['nationality'], prefix='nat', drop_first=True),
        pd.get_dummies(df['track_type'], prefix='tracktype', drop_first=True)
    ], axis=1)
    y = df['position']
    print(f"✅ Feature matrix created with shape: {X.shape}")
    return X, y


def train_model(X, y):
    """Trains the XGBoost model using GridSearchCV."""
    print("🧠 Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = XGBRegressor(random_state=42)
    param_grid = {'n_estimators': [200, 400], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(base_model, param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_rmse = -grid.best_score_

    print(f"✅ Model trained! 🎯 Final RMSE: {final_rmse:.2f}")
    print(f"✅ Best CV (5-fold) RMSE: {cv_rmse:.2f}")
    return model, X, final_rmse, cv_rmse


def save_artifacts(model, X, final_rmse, cv_rmse, engineered_df):
    """Saves all necessary model artifacts."""
    print("💾 Saving artifacts...")

    joblib.dump(model, "model_xgb.pkl")

    model_features = pd.DataFrame(X.columns, columns=['feature'])
    model_features.to_csv("model_features.csv", index=False)

    metrics = {"rmse": float(final_rmse), "cv_rmse": float(cv_rmse), "features": int(X.shape[1])}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📊 Generating SHAP plot...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X.iloc[:200])
    plt.figure()
    shap.summary_plot(shap_values, X.iloc[:200], show=False)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight')
    plt.close()

    print("📊 Calculating track advantage scores...")
    track_advantage = engineered_df.groupby('name')['position_change'].mean().reset_index()
    track_advantage.rename(columns={'name': 'track', 'position_change': 'avg_position_gain'}, inplace=True)
    track_advantage.to_csv('track_advantage_scores.csv', index=False)

    print("✅ All artifacts saved successfully.")


def main():
    """Main function to run the entire ML pipeline."""
    data = load_data()
    if data:
        races, results, qualifying, drivers, constructors = data
        engineered_df = engineer_features(races, results, qualifying, drivers, constructors)
        X, y = create_feature_matrix(engineered_df)
        model, X, final_rmse, cv_rmse = train_model(X, y)
        save_artifacts(model, X, final_rmse, cv_rmse, engineered_df)
        print("\n🎉 F1 Project pipeline finished successfully! 🎉")


if __name__ == "__main__":
    main()