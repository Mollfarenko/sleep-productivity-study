import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

def train_logistic_regression_pipeline():
    start_time = time.time()  # Start timer

    # Load preprocessed data
    data_path = "data/processed/sleep_productivity_classification.csv"
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop(columns=["Productivity Class"])  # Assuming target column is "Productivity Binary"
    y = df["Productivity Class"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline with scaling + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Define hyperparameter grid
    param_grid = [
    {
        "classifier__penalty": ["l1"],
        "classifier__solver": ["liblinear"],
        "classifier__C": [0.01, 0.1, 1, 10]
    },
    {
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs"],
        "classifier__C": [0.01, 0.1, 1, 10]
    }
    ]

    # K-Fold Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search for best parameters
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",  # You could also use 'f1' or 'roc_auc' depending on context
        n_jobs=-1,
        verbose=1
    )

    # Start MLflow experiment
    mlflow.set_experiment("Sleep Productivity Classification")

    with mlflow.start_run(run_name="Tuned Logistic Regression Model"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("Model", "TunedLogisticRegression")
        mlflow.log_param("best_params", grid_search.best_params_)
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("ROC AUC", roc_auc)

        # Save model locally
        model_dir = "models/tuned_logistic_regression"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "tuned_logistic_regression.pkl")
        joblib.dump(best_model, model_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(best_model, "tuned_logistic_regression")

        print(f"Model logged in MLflow: Accuracy={acc:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

    end_time = time.time()  # End timer
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    return best_model

if __name__ == "__main__":
    train_logistic_regression_pipeline()
