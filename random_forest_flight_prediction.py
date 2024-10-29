#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('sfoMayNonstop.csv')

# Preprocess the data
# Select relevant columns
df = df[['searchDate', 'destinationAirport', 'isNonStop', 'seatsRemaining', 'totalTravelDistance', 'baseFare']]

# Handle missing values
df = df.dropna()

# Feature Engineering
# Convert 'searchDate' to datetime format and extract day, month, and year as separate features
df['searchDate'] = pd.to_datetime(df['searchDate'])
df['day'] = df['searchDate'].dt.day
df['month'] = df['searchDate'].dt.month
df['year'] = df['searchDate'].dt.year
df['isWeekend'] = (df['searchDate'].dt.dayofweek >= 5).astype(int)

# Encode categorical features
df['isNonStop'] = df['isNonStop'].astype(int)

# Create target variable by categorizing baseFare
# Defining price categories: Low, Medium, High based on baseFare quantiles
df['priceCategory'] = pd.qcut(df['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Define target variable and feature set
X = df[['seatsRemaining', 'totalTravelDistance', 'isNonStop', 'isWeekend', 'day', 'month', 'year']]
y = df['priceCategory']

# Split data into training (80%) and test (20%) sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into 60% training and 20% validation (of the original dataset)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

print("Dataset Sizes:")
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Initialize Random Forest Classifier with initial hyperparameters
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_rf_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train the model with the best parameters on the full training data (train + validation)
best_rf_model.fit(X_train_full, y_train_full)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)

# Evaluate the model on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test, drop_first=False), y_pred_proba, average='weighted', multi_class='ovr')

# Print evaluation metrics and detailed classification report
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot feature importances
feature_importances = best_rf_model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances in Random Forest Classifier")
plt.show()

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Fine-tuning analysis: Plot accuracy, precision, recall, and f1 for each fold
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values(by="rank_test_score")
metrics = ['mean_test_score', 'mean_fit_time']

# Plot each metric across parameters
for metric in metrics:
    plt.figure()
    sns.lineplot(data=cv_results, x='param_n_estimators', y=metric, hue='param_max_depth')
    plt.title(f'{metric} vs Number of Estimators by Max Depth')
    plt.xlabel('Number of Estimators')
    plt.ylabel(metric)
    plt.legend(title="Max Depth")
    plt.show()

# Plot performance on validation set vs training set
train_accuracies = []
val_accuracies = []
num_trees_range = range(10, 201, 10)

for n in num_trees_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train, y_train)
    train_accuracies.append(rf_temp.score(X_train, y_train))
    val_accuracies.append(rf_temp.score(X_val, y_val))

plt.plot(num_trees_range, train_accuracies, label='Train Accuracy')
plt.plot(num_trees_range, val_accuracies, label='Validation Accuracy')
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Train vs Validation Accuracy with Varying Trees")
plt.show()
