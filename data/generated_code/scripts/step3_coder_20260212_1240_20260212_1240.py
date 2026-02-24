import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Ensure the directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic dataset for demonstration
np.random.seed(42)
n_samples = 1000

# Create features: age, income, education years
age = np.random.randint(18, 80, n_samples)
income = np.random.normal(50000, 20000, n_samples)
education_years = np.random.randint(8, 20, n_samples)

# Create target variable: salary based on features with some noise
salary = (age * 1000 + 
          income * 0.8 + 
          education_years * 5000 + 
          np.random.normal(0, 5000, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'income': income,
    'education_years': education_years,
    'salary': salary
})

# Save dataset to CSV
data.to_csv(f'{output_dir}/salary_data.csv', index=False)

# Display first few rows of data
print("Dataset sample:")
print(data.head())

# Prepare features and target
X = data[['age', 'income', 'education_years']]
y = data['salary']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling to normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Create visualization of predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Linear Regression: Actual vs Predicted Salary')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig(f'{output_dir}/actual_vs_predicted_salary.png')
plt.close()

# Feature importance visualization (coefficients)
feature_names = ['Age', 'Income', 'Education Years']
coefficients = model.coef_

plt.figure(figsize=(10, 6))
bars = plt.bar(feature_names, coefficients, color=['red', 'green', 'blue'])
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance in Salary Prediction')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, coef in zip(bars, coefficients):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(coefficients) - min(coefficients))*0.01,
             f'{coef:.0f}', ha='center', va='bottom')

# Save the feature importance plot
plt.savefig(f'{output_dir}/feature_importance.png')
plt.close()

# Create a prediction example
example_age = 35
example_income = 60000
example_education = 16

# Prepare example data
example_data = np.array([[example_age, example_income, example_education]])
example_scaled = scaler.transform(example_data)

# Make prediction
predicted_salary = model.predict(example_scaled)[0]
print(f"\nPrediction Example:")
print(f"Person with age {example_age}, income ${example_income:,}, education {example_education} years")
print(f"Predicted salary: ${predicted_salary:,.2f}")

# Print model equation
print(f"\nModel Equation:")
print(f"Salary = {model.intercept_:.2f} + Age × {model.coef_[0]:.2f} + Income × {model.coef_[1]:.2f} + Education × {model.coef_[2]:.2f}")

print(f"\nAll visualizations saved to {output_dir}")