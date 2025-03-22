import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_dummy_classifiers():
    # Load preprocessed data
    df = pd.read_csv("data/processed/sleep_productivity_classification.csv")

    # Features and target
    X = df.drop(columns=["Productivity Class"])  # All features except target
    y = df["Productivity Class"]  # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # List of strategies to test
    strategies = ["most_frequent", "stratified"]

    mlflow.set_experiment("Sleep Productivity Dummies")

    # Run dummy classifiers with MLflow
    for strategy in strategies:
        with mlflow.start_run(run_name=f"DummyClassifier-{strategy}"):
            model = DummyClassifier(strategy=strategy)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log metrics
            mlflow.log_param("strategy", strategy)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("F1 Score", f1)

            print(f"Strategy: {strategy}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")

if __name__ == "__main__":
    train_dummy_classifiers()
