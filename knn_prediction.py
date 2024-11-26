import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Load the dataset
file_path = 'flights.csv'
data = pd.read_csv(file_path)

columns_to_drop = ['legId', 'fareBasisCode', 'segmentsDepartureTimeEpochSeconds', 
                   'segmentsArrivalTimeEpochSeconds', 'segmentsDepartureTimeRaw', 'segmentsArrivalTimeRaw',
                   'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode', 'segmentsAirlineName', 
                   'segmentsAirlineCode', 'segmentsEquipmentDescription', 'segmentsDurationInSeconds', 
                   'segmentsDistance', 'segmentsCabinCode']

data.drop(columns=columns_to_drop, axis=1, inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Convert 'searchDate' to datetime and extract day of week
data['searchDate'] = pd.to_datetime(data['searchDate'])
data['searchDayOfWeek'] = data['searchDate'].dt.dayofweek

# Calculate days between search and flight date
data['daysBeforeFlight'] = (pd.to_datetime('2022-05-31') - data['searchDate']).dt.days

# Define target variable by categorizing 'baseFare' into Low, Medium, High
data['priceCategory'] = pd.qcut(data['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Define feature set and target variable
X = data[['seatsRemaining', 'totalTravelDistance', 'searchDayOfWeek', 'daysBeforeFlight', 'isBasicEconomy']]
y = data['priceCategory']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define the range for k values
param_grid = {'n_neighbors': range(1, 21)}

knn_model = KNeighborsClassifier()

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Print the best k value and corresponding score
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best F1 Score:", grid_search.best_score_)

# Extending the range of k values
param_grid = {'n_neighbors': range(21, 51)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best F1 Score:", grid_search.best_score_)

# Train the KNN model with the optimal k=42
best_knn_model = KNeighborsClassifier(n_neighbors=42)
best_knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_knn_model.predict(X_test)
y_pred_proba = best_knn_model.predict_proba(X_test)

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test, drop_first=False), y_pred_proba, average='weighted', multi_class='ovr')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for KNN with k=42")

output_file = "confusion_matrix_knn_k42.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

plt.show()

# Evaluate permutation importance
result = permutation_importance(best_knn_model, X_test, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean
print("Feature importance:", importance)
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Permutation Feature Importances for KNN Classifier")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()

# Save the plot
output_file = "knn_permutation_feature_importances.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
