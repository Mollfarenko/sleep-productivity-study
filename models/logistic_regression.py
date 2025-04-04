import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

def train_logistic_regression():
    start_time = time.time()  # Start timer

    # Load preprocessed data
    data_path = "data/processed/sleep_productivity_classification.csv"
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop(columns=["Productivity Class"])  # Assuming target column is "Productivity Binary"
    y = df["Productivity Class"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Start MLflow experiment
    mlflow.set_experiment("Sleep Productivity Classification")

    with mlflow.start_run(run_name="Logistic Regression Model"):
        # Initialize and train model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Model", "LogisticRegression")
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("ROC AUC", roc_auc)

        # Save model locally
        model_dir = "models/logistic_regression"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "logistic_regression.pkl")
        joblib.dump(model, model_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(model, "logistic_regression")

        print(f"Model logged in MLflow: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

    end_time = time.time()  # End timer
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    return model

if __name__ == "__main__":
    train_logistic_regression()
