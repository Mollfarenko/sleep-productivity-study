# 💤 Sleep Productivity Prediction Project
This project explores the relationship between sleep habits and productivity using various machine learning techniques. The goal was to predict productivity scores based on features like sleep duration, quality, and other lifestyle habits.

However, upon closer inspection, it became clear that the dataset was simulated and largely random, limiting its potential to yield meaningful insights. Despite this, the project served as an excellent opportunity to practice end-to-end data science workflows, including preprocessing, model building, evaluation, and experiment tracking with MLflow.

## 📊 Dataset

- Simulated sleep dataset with features like:
  - Sleep Duration
  - Sleep Quality
  - Physical Activity
  - Mood
  - Caffeine intake
  - Productivity score


## 🧪 Models Used

- Dummy Regressors & Classifiers (Baselines)
- Linear Regression
- Random Forest Regressor
- Logistic Regression for Classification

## 🧼 Preprocessing

- Feature extraction from datetime
- Gender encoding
- Scaling via StandardScaler
- Binary classification prep

## 🔬 Evaluation Metrics

- Regression: MSE, R²
- Classification: Accuracy, F1 score

## 📈 Results

- Best Regression Model: **Linear Regression** (MSE 8.07)
- Best Classifier: **Logistic Regression** and **Dummy Classifier** (Accuracy 58%)

Visual reports available in the [`reports/`](./reports) folder.

## 📦 MLflow Tracking

All experiments and model metrics logged using MLflow.  
To run MLflow UI

