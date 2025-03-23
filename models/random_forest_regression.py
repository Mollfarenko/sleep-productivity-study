import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

    # Initialize Random Forest (default parameters)
    model = RandomForestRegressor(random_state=42)

    # Start MLflow experiment
    mlflow.set_experiment("Sleep Productivity Regression Model")

    with mlflow.start_run(run_name="Random Forest Regressor"):
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log parameters & metrics in MLflow
        mlflow.log_param("Model", "RandomForestRegressor")
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Save model
        model_path = "models/random_forest"
        os.makedirs(model_path, exist_ok=True)
        mlflow.sklearn.log_model(model, model_path)

        print(f"Model logged in MLflow: MSE={mse:.4f}, R²={r2:.4f}")

    # End timer
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    return model  # Return trained model for further use

if __name__ == "__main__":
    train_random_forest()
