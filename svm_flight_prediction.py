#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'flights.csv'
data = pd.read_csv(file_path)

# Preprocessing the data
# Select relevant columns for consistency with teammates' work
data = data[['searchDate', 'isNonStop', 'seatsRemaining', 'totalTravelDistance', 'baseFare']]

# Handle missing values
data.dropna(inplace=True)

# Feature Engineering
# Convert 'searchDate' to datetime and extract day of week
data['searchDate'] = pd.to_datetime(data['searchDate'])
data['searchDayOfWeek'] = data['searchDate'].dt.dayofweek

# Calculate days between search and flight date
data['daysBeforeFlight'] = (pd.to_datetime('2022-05-31') - data['searchDate']).dt.days

# Convert categorical features
data['isNonStop'] = data['isNonStop'].astype(int)

# Define target variable by categorizing 'baseFare' into Low, Medium, High
data['priceCategory'] = pd.qcut(data['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Define feature set and target variable
X = data[['seatsRemaining', 'totalTravelDistance', 'isNonStop', 'searchDayOfWeek', 'daysBeforeFlight']]
y = data['priceCategory']

# Standardize features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:


# Initialize SVM model with initial parameters
svm_model = SVC(kernel='linear', C=1, decision_function_shape='ovr', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)


# In[ ]:


# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test, drop_first=False), y_pred_proba, average='weighted', multi_class='ovr')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM")
plt.show()


# In[ ]:


# Optional: Hyperparameter Tuning with Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(decision_function_shape='ovr', probability=True), param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Display best hyperparameters from Grid Search
print("Best Hyperparameters from Grid Search:", grid_search.best_params_)

# Train with best parameters from Grid Search
best_svm = grid_search.best_estimator_
best_svm.fit(X_train, y_train)

# Make predictions with tuned model
y_pred_best = best_svm.predict(X_test)
y_pred_proba_best = best_svm.predict_proba(X_test)

# Evaluate tuned model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='weighted')
recall_best = recall_score(y_test, y_pred_best, average='weighted')
f1_best = f1_score(y_test, y_pred_best, average='weighted')
roc_auc_best = roc_auc_score(pd.get_dummies(y_test, drop_first=False), y_pred_proba_best, average='weighted', multi_class='ovr')

print(f'\nTuned Model - Accuracy: {accuracy_best:.2f}')
print(f'Tuned Model - Precision: {precision_best:.2f}')
print(f'Tuned Model - Recall: {recall_best:.2f}')
print(f'Tuned Model - F1 Score: {f1_best:.2f}')
print(f'Tuned Model - ROC AUC Score: {roc_auc_best:.2f}')
print("\nTuned Model Classification Report:\n", classification_report(y_test, y_pred_best))


# In[ ]:




