# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('flights.csv')

# Preprocess the data
# Handle missing values by dropping rows with NaN
df = df.dropna()

# Feature Engineering
# Convert 'searchDate' to datetime format
df['searchDate'] = pd.to_datetime(df['searchDate'])

# Create isWeekend feature
df['isWeekend'] = (df['searchDate'].dt.dayofweek >= 5).astype(int)  # 1 for Saturday/Sunday, 0 otherwise

# Extract day, month, and year from searchDate
df['day'] = df['searchDate'].dt.day
df['month'] = df['searchDate'].dt.month
df['year'] = df['searchDate'].dt.year

# Extract searchDayOfWeek feature (0 = Monday, ..., 6 = Sunday)
df['searchDayOfWeek'] = df['searchDate'].dt.dayofweek

# Calculate daysBeforeFlight if 'flightDate' column exists
if 'flightDate' in df.columns:
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    df['daysBeforeFlight'] = (df['flightDate'] - df['searchDate']).dt.days
else:
    df['daysBeforeFlight'] = np.nan  # Placeholder if flightDate is not present

# Fill missing values in daysBeforeFlight with 0 (if any)
df['daysBeforeFlight'] = df['daysBeforeFlight'].fillna(0)

# Encode 'isBasicEconomy' as an integer
df['isBasicEconomy'] = df['isBasicEconomy'].astype(int)

# Create target variable (priceCategory) based on quantiles of 'baseFare'
df['priceCategory'] = pd.qcut(df['baseFare'], q=3, labels=['Low', 'Medium', 'High'])

# Define feature set and target variable with the specified features
X = df[['seatsRemaining', 'isBasicEconomy', 'totalTravelDistance', 'searchDayOfWeek', 'daysBeforeFlight']]
y = df['priceCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Visualization with Different Colors
feature_importances = rf_model.feature_importances_
features = X.columns

# Create a color palette for the bars
palette = sns.color_palette("husl", len(features))  # 'husl' generates distinct colors for each feature

sns.barplot(x=feature_importances, y=features, palette=palette)
plt.title("Feature Importances in Random Forest Classifier")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
