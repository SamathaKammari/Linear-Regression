#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[2]:


# Load the dataset
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')


# In[3]:


data.head()


# In[4]:


# 1. Preprocess data
# Handle categorical variable 'gender'
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})


# In[5]:


# Check for missing values in all columns
print("Missing values before handling:\n", data.isnull().sum())


# In[6]:


# Separate features and target
X = data.drop(['is_patient'], axis=1)  # All features for multiple regression
X_simple = data[['age']]               # Single feature for simple regression
y = data['is_patient']


# In[7]:


# Impute missing values for all numerical columns
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_simple = pd.DataFrame(imputer.fit_transform(X_simple), columns=X_simple.columns)


# In[8]:


# Verify no missing values remain
print("\nMissing values after handling (X):\n", X.isnull().sum())
print("\nMissing values after handling (X_simple):\n", X_simple.isnull().sum())


# In[9]:


# Scale features for multiple regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[10]:


# 2. Split data into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_simple, X_test_simple, _, _ = train_test_split(X_simple, y, test_size=0.2, random_state=42)


# In[11]:


# 3. Fit Linear Regression models
# Simple Linear Regression (using 'age')
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train.values.reshape(-1, 1))

# Multiple Linear Regression
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)


# In[18]:


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"\n{model_name} Evaluation:")
    print("Training Set:")
    print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"R²: {r2_score(y_train, y_train_pred):.4f}")
    
    print("\nTest Set:")
    print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_test_pred):.4f}")

# Evaluate both models
evaluate_model(model_simple, X_train_simple, X_test_simple, y_train, y_test, "Simple Linear Regression (Age)")
evaluate_model(model_multiple, X_train, X_test, y_train, y_test, "Multiple Linear Regression")


# In[15]:


# 5. Plot regression line (for simple linear regression with 'age')
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test, color='blue', label='Test data')
X_range = np.linspace(X_test_simple.min(), X_test_simple.max(), 100).reshape(-1, 1)
y_range_pred = model_simple.predict(X_range)
plt.plot(X_range, y_range_pred, color='red', label='Regression line')
plt.xlabel('Age')
plt.ylabel('Is Patient (1: Yes, 2: No)')
plt.title('Simple Linear Regression: Age vs Liver Patient Status')
plt.legend()
plt.savefig('simple_linear_regression_ilpd.png')
plt.show()
plt.close()


# In[16]:


# 6. Interpret coefficients
print("\nSimple Linear Regression Coefficients:")
print(f"Intercept: {float(model_simple.intercept_):.4f}")  # Convert intercept_ to float
print(f"Coefficient for Age: {model_simple.coef_[0].item():.4f}")  # Use .item() for coef_
print(f"Interpretation: For each additional year of age, the predicted patient status changes by {model_simple.coef_[0].item():.4f} (1: patient, 2: non-patient).")

print("\nMultiple Linear Regression Coefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': [float(coef) for coef in model_multiple.coef_]  # Convert coefficients to float
})
print(coef_df)
print(f"Intercept: {float(model_multiple.intercept_):.4f}")  # Convert intercept_ to float
print("Interpretation: Each coefficient represents the change in predicted patient status for a one-unit increase in the standardized feature, holding others constant.")


# In[ ]:




