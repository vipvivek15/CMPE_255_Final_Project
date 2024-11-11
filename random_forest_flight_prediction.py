# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('flights.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Dataset Sizes:")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Initialize Random Forest Classifier with the best hyperparameters
# No need to apply grid search since it was already used to fetch the best hyperparameters
# Best parameters: {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced'
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

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
feature_importances = rf_model.feature_importances_
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
