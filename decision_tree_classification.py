# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
filePath = 'flights.csv'
data = pd.read_csv(filePath)

# Preprocessing the data
# Feature Selection
# Drop columns that are not necessary for the classification task
columns_to_drop = ['legId', 'fareBasisCode', 'segmentsDepartureTimeEpochSeconds', 
                   'segmentsArrivalTimeEpochSeconds', 'segmentsDepartureTimeRaw', 'segmentsArrivalTimeRaw',
                   'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode', 'segmentsAirlineName', 
                   'segmentsAirlineCode', 'segmentsEquipmentDescription', 'segmentsDurationInSeconds', 
                   'segmentsDistance', 'segmentsCabinCode']

data.drop(columns=columns_to_drop, axis=1, inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Convert 'searchDate' to datetime format and extract day, month, and year as separate features
data['searchDate'] = pd.to_datetime(data['searchDate'])
data['flightDate'] = pd.to_datetime(data['flightDate'])

# Get day of the week (0 = Monday, 6 = Sunday)
data['searchDayOfWeek'] = data['searchDate'].dt.dayofweek

# Calculate days between search and flight date
data['daysBeforeFlight'] = (data['flightDate'] - data['searchDate']).dt.days

# Encode categorical features
data['isBasicEconomy'] = data['isBasicEconomy'].astype(int)

# Defining price categories: Low, Medium, High using binning
data['priceCategory'] = pd.qcut(data['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Selecting features and target
X = data[['seatsRemaining', 'isBasicEconomy', 'totalTravelDistance', 'searchDayOfWeek', 'daysBeforeFlight']]
y = data['priceCategory']

# Split data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset Sizes:")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Decision Tree classifier

# Grid Search
# classifier = DecisionTreeClassifier()
# params={'criterion':['gini', 'entropy'],  
#         'max_depth': [5, 10, 15, None], 
#         'min_samples_split': [2, 5, 10, 15],
#         'min_samples_leaf': [1, 2, 4, 5] 
#         }

# gs=GridSearchCV(estimator=classifier, param_grid=params, cv=5, scoring='accuracy')   
# gs = gs.fit(X_train, y_train)
# print("Best Parameters:", gs.best_params_)

# After performing grid search the best hyperparameters are:
# 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 15
classifier = DecisionTreeClassifier(min_samples_leaf = 5, min_samples_split=15, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics and detailed classification report
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot feature importances
feature_importances = classifier.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances in Decision Tree Classifier")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Decision Tree")
plt.show()
