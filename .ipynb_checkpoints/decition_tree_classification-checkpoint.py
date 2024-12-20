# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Defining price categories: Low, Medium, High based on baseFare quantiles
data['priceCategory'] = pd.qcut(data['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Selecting features and target
X = data[['seatsRemaining', 'totalTravelDistance', 'searchDayOfWeek', 'daysBeforeFlight']]
y = data['priceCategory']

# Split data into training (80%) and test (20%) sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into 60% training and 20% validation (of the original dataset)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

print("Dataset Sizes:")
print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model on test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics and detailed classification report
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))