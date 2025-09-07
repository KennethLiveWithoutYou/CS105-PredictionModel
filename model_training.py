import pandas as pd
import pickle
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np

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