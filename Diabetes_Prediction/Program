#Diabetes_prediction

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="DiseaseProgression")

print("First 5 rows of dataset:")
print(X.head())

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train models
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

#Predictions
y_pred_lr = lin_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

#Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    return {"Model": model_name, "MSE": mse, "MAE": mae, "R2": r2}

#Evaluate both models
results = []
results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))

#Compare results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

#Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred_lr, label="Linear Regression", marker='x')
plt.plot(y_pred_rf, label="Random Forest", marker='s')
plt.title("Actual vs Predicted Disease Progression")
plt.xlabel("Sample Index")
plt.ylabel("Progression Score")
plt.legend()
plt.show()

#Heatmap of feature correlation
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
