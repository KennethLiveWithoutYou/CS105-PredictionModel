import pandas as pd
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import os

os.makedirs("static/models", exist_ok=True)

def save_regression_plots(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.savefig(f"static/models/{model_name}_actual_vs_pred.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, standardized_residuals, alpha=0.7)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Standardized Residuals")
    plt.title(f"{model_name} - Residual Plot")
    plt.savefig(f"static/models/{model_name}_residuals.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.hist(standardized_residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Standardized Residuals")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} - Residual Histogram")
    plt.savefig(f"static/models/{model_name}_residual_hist.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    stats.probplot(standardized_residuals, dist="norm", plot=plt)
    plt.title(f"{model_name} - QQ Plot of Residuals")
    plt.savefig(f"static/models/{model_name}_qqplot.png")
    plt.close()

def train_and_save_model(X, y, model_path, metrics_path, corr_path=None, selected_features=None):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Using 5-Fold CV for dataset with {len(X)} rows")

    cv_r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    cv_preds = cross_val_predict(pipe, X, y, cv=cv)

    metrics = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "cv_r2_scores": cv_r2_scores.tolist(),
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "cv_rmse": mean_squared_error(y, cv_preds) ** 0.5
    }
    print(f"Metrics for {model_path}: {metrics}")

    pipe.fit(X, y)
    pickle.dump(pipe, open(model_path, "wb"))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if corr_path:
        with open(corr_path, "wb") as f:
            pickle.dump(selected_features, f)

    model_name = model_path.split("/")[-1].replace(".pkl", "")
    save_regression_plots(y, cv_preds, model_name)

    return pipe

# SALARY DATA
salary_data = pd.read_csv("data/salary_data.csv")
print(f"Total missing values: {salary_data.isnull().sum().sum()}")

salary_data_X = salary_data[["YearsExperience"]]
salary_data_y = salary_data["Salary"]

train_and_save_model(
    salary_data_X,
    salary_data_y,
    model_path="models/salary_model.pkl",
    metrics_path="models/salary_metrics.json"
)

# INCOME DATA
income_data = pd.read_csv("data/income_dataset.csv")
print(f"Total missing values: {income_data.isnull().sum().sum()}")

income_data_X = income_data[["age", "experience"]]
income_data_y = income_data["income"]

train_and_save_model(
    income_data_X,
    income_data_y,
    model_path="models/income_model.pkl",
    metrics_path="models/income_metrics.json"
)

# ADVERTISING DATA
advertise_data = pd.read_csv("data/advertising-dataset.csv")
print(f"Total missing values: {advertise_data.isnull().sum().sum()}")

correlations = advertise_data.corr()
print(f"Correlation matrix:\n{correlations}")

corr_on_y = correlations["Sales"].drop("Sales")
print(f"\nCorrelation on Sales:\n{corr_on_y}")

corr_dict = corr_on_y.to_dict()

with open("models/advertise_corr.pkl", "wb") as f:
    pickle.dump(corr_dict, f)

selected_features = corr_on_y[corr_on_y.abs() >= 0.3].index.tolist()
print(f"Selected features: {selected_features}")

advertise_data_X = advertise_data[selected_features]
advertise_data_y = advertise_data["Sales"]

train_and_save_model(
    advertise_data_X,
    advertise_data_y,
    model_path="models/advertise_model.pkl",
    metrics_path="models/advertise_metrics.json",
    corr_path="models/advertise_features.pkl",
    selected_features=selected_features
)