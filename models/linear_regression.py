import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression():
    start_time = time.time()  # Start timer

    # Load preprocessed data
    data_path = "data/processed/sleep_productivity_clean.csv"
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop(columns=["Productivity Score"])
    y = df["Productivity Score"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline: Scaling + Linear Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # Normalize features
        ("regressor", LinearRegression())  # Placeholder for GridSearch
    ])

    # Define hyperparameter grid for tuning
    param_grid = [
        {"regressor": [LinearRegression()]},  # No alpha for LinearRegression
        {"regressor": [Ridge(), Lasso()], "regressor__alpha": [0.01, 0.1, 1, 10, 100]}  # Alpha only for Ridge & Lasso
    ]

    # GridSearchCV to find best model & parameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

    # Start MLflow experiment
    mlflow.set_experiment("Sleep Productivity Regression Model")

    with mlflow.start_run(run_name="Linear Regression Model Pipeline"):
        # Train model
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log best model & parameters in MLflow
        mlflow.log_param("Best Model", str(grid_search.best_params_["regressor"]))
        if "regressor__alpha" in grid_search.best_params_:
            mlflow.log_param("Best Alpha", grid_search.best_params_["regressor__alpha"])
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Save model
        model_path = "models/linear_regression_pipeline"
        os.makedirs(model_path, exist_ok=True)
        mlflow.sklearn.log_model(best_model, model_path)

        print(f"Best Model: {grid_search.best_params_['regressor']}")
        print(f"Best Alpha: {grid_search.best_params_.get('regressor__alpha', 'N/A')}")
        print(f"Model logged in MLflow: MSE={mse:.4f}, R²={r2:.4f}")

    # End timer
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    return best_model  # Return trained model for further use

if __name__ == "__main__":
    train_linear_regression()

