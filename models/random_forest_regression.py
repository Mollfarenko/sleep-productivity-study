import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.pipeline import Pipeline


def train_random_forest():
    start_time = time.time()  # Start timer

    # Load preprocessed data
    data_path = "data/processed/sleep_productivity_clean.csv"
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop(columns=["Productivity Score"])
    y = df["Productivity Score"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline: Scaling (optional) + Random Forest
    rf_pipeline = Pipeline([
        #("scaler", StandardScaler()),  # Not strictly necessary for RandomForest
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    # Define hyperparameter grid for tuning
    param_grid = {
        "regressor__n_estimators": [50, 100, 200, 500],
        "regressor__max_depth": [None, 10, 20, 30],
        "regressor__min_samples_split": [2, 5, 10]
    }

    # GridSearchCV to find best model & parameters
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)

    # Start MLflow experiment
    mlflow.set_experiment("Sleep Productivity Regression Model")

    with mlflow.start_run(run_name="Random Forest Regressor Pipeline"):
        # Train model with Grid Search
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log best model & parameters in MLflow
        mlflow.log_param("Best Model", "RandomForestRegressor")
        mlflow.log_param("Best n_estimators", grid_search.best_params_["regressor__n_estimators"])
        mlflow.log_param("Best max_depth", grid_search.best_params_["regressor__max_depth"])
        mlflow.log_param("Best min_samples_split", grid_search.best_params_["regressor__min_samples_split"])
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Save model locally
        model_dir = "models/random_forest"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest.pkl")
        joblib.dump(best_model, model_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(best_model, artifact_path="random_forest_model")

        print(f"Best Model: {grid_search.best_params_}")
        print(f"Model saved locally at {model_path}")
        print(f"Model logged in MLflow: MSE={mse:.4f}, R²={r2:.4f}")

    elapsed_time = time.time() - start_time  # Stop timer
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    return best_model  # Return trained model for further use

if __name__ == "__main__":
    train_random_forest()

