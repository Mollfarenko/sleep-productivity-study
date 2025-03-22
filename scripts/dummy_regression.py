import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_dummy_regressors():
    # Load preprocessed data
    df = pd.read_csv("data/processed/sleep_productivity_clean.csv")

    # Define features (X) and target (y)
    X = df.drop(columns=["Productivity Score"])
    y = df["Productivity Score"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define different strategies
    strategies = ["mean", "median", "constant", "uniform"]
    constant_value = 5.0   # Example constant value

    mlflow.set_experiment("Sleep Productivity Dummies")

    for strategy in strategies:
        with mlflow.start_run(run_name=f"DummyRegressor-{strategy}"):
            # Handle specific parameters for constant
            params = {"strategy": strategy}
            if strategy == "constant":
                params["constant"] = constant_value
                dummy_model = DummyRegressor(strategy=strategy, constant=constant_value)
            elif strategy == "uniform":
                # Create a random baseline model (not using DummyRegressor)
                y_pred_random = np.random.uniform(y_train.min(), y_train.max(), size=len(y_test))
                # Evaluate model
                mse = mean_squared_error(y_test, y_pred_random)
                # Log results in MLflow
                mlflow.log_metric("MSE", mse)
            else:
                dummy_model = DummyRegressor(strategy=strategy)

            # Train model
            dummy_model.fit(X_train, y_train)

            # Predictions
            y_pred = dummy_model.predict(X_test)

            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)

            # Log results in MLflow
            mlflow.log_params(params)
            mlflow.log_metric("MSE", mse)
            mlflow.sklearn.log_model(dummy_model, f"dummy_model_{strategy}")

            print(f"Strategy: {strategy} | MSE: {mse:.2f}")

    print("All dummy models logged in MLflow.")

if __name__ == "__main__":
    train_dummy_regressors()
