import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SALARY DATA
salary_data = pd.read_csv("data/salary_data.csv")

print(f"Total missing values: {salary_data.isnull().sum().sum()}")

salary_data_X = salary_data[["YearsExperience"]]
salary_data_y = salary_data["Salary"]

salary_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)
salary_pipe.fit(salary_data_X, salary_data_y)

pickle.dump(salary_pipe, open("models/salary_model.pkl", "wb"))

# INCOME DATA
income_data = pd.read_csv("data/income_dataset.csv")

print(f"Total missing values: {income_data.isnull().sum().sum()}", )

income_data_X = income_data[["age", "experience"]]
income_data_y = income_data["income"]

income_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)
income_pipe.fit(income_data_X, income_data_y)

pickle.dump(income_pipe, open("models/income_model.pkl", "wb"))

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

advertise_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]
)
advertise_pipe.fit(advertise_data_X, advertise_data_y)

pickle.dump(advertise_pipe, open("models/advertise_model.pkl", "wb"))