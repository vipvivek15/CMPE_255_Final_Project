#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('sfoMayNonstop.csv')

# Preprocess the data
# Select relevant columns based on corrected column names
df = df[['searchDate', 'destinationAirport', 'isNonStop', 'seatsRemaining', 'totalTravelDistance', 'baseFare']]

# Handle missing values if any
df = df.dropna()

# Feature Engineering
# Convert date columns to datetime format if they aren't already
df['searchDate'] = pd.to_datetime(df['searchDate'])

# Extract new features from date (e.g., weekday/weekend indicator)
df['isWeekend'] = df['searchDate'].dt.dayofweek >= 5

# Encode categorical features
df['isNonStop'] = df['isNonStop'].astype(int)  # assuming non-stop is binary
df['isWeekend'] = df['isWeekend'].astype(int)

# Create target variable by categorizing baseFare
# Defining price categories: Low, Medium, High based on baseFare quantiles
df['priceCategory'] = pd.qcut(df['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Define target variable and feature set
X = df[['seatsRemaining', 'totalTravelDistance', 'isNonStop', 'isWeekend']]
y = df['priceCategory']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Plot feature importance
feature_importances = rf_model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances in Random Forest Classifier")
plt.show()

# Visualize model performance
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

