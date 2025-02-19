import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

X = df.drop(columns=["Target"])
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressors = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Bayesian Ridge": BayesianRidge(),
    "Elastic Net": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "KNN": KNeighborsRegressor()
}

results = []
for name, model in regressors.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append([name, mae, mse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "R²"]).sort_values(by="R²", ascending=False)
print(results_df)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="R²", y="Model", palette="viridis")
plt.xlabel("R² Score")
plt.ylabel("Regression Model")
plt.title("Comparison of Regression Models")
plt.show()
