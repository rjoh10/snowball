# Goal: To create a logistic regression model that predicts whether a wildfire
# will be large or small based on the cause of the fire. The independent variable
# will be the cause of the fire, and the dependent variable will be the fire size,
# using the GIS Calculated Acres column to determine if a fire is classified as
# large or small.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the original dataset
file_path = '/Users/rheajohnson/Downloads/caFirePerimeters.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns if they exist
columns_to_drop = [
    'State', 'Alarm Date', 'Containment Date', 'Complex Name', 'Fire Number (historical use)',
    'Agency', 'Fire ID', 'Unit ID', 'Fire Name', 'Local Incident Number', 'Date',
    'Comments', 'IRWIN ID', 'Complex ID'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Map numeric causes to categorical labels
cause_mapping = {
    1: 'Lightning', 2: 'Equipment Use', 3: 'Smoking', 4: 'Campfire', 5: 'Debris',
    6: 'Railroad', 7: 'Arson', 8: 'Playing with fire', 9: 'Miscellaneous', 10: 'Vehicle Accident',
    11: 'Powerline', 12: 'Firefighter Training', 13: 'Non-Firefighter Training',
    14: 'Unknown/Unidentified', 15: 'Structure', 16: 'Aircraft', 17: 'Volcanic',
    18: 'Escaped Prescribed Burn', 19: 'Illegal Alien Campfire'
}
df['Cause'] = df['Cause'].map(cause_mapping)

# Human and Nature Sorting
human_causes = [
    'Smoking', 'Campfire', 'Equipment Use', 'Playing with fire', 'Arson', 'Railroad',
    'Vehicle', 'Powerline', 'Firefighter Training', 'Non-Firefighter Training',
    'Structure', 'Aircraft', 'Escaped Prescribed Burn', 'Illegal Alien Campfire'
]
nature_causes = ['Lightning', 'Volcanic', 'Debris']
df['Cause_Category'] = df['Cause'].apply(lambda x: 'Human' if x in human_causes else 'Nature' if x in nature_causes else np.nan)

# Clean missing data in numeric columns
df['GIS Calculated Acres'] = pd.to_numeric(df['GIS Calculated Acres'], errors='coerce').fillna(df['GIS Calculated Acres'].mean())
df['Shape__Length'] = pd.to_numeric(df['Shape__Length'], errors='coerce').fillna(df['Shape__Length'].mean())
df['Shape__Area'] = pd.to_numeric(df['Shape__Area'], errors='coerce').fillna(df['Shape__Area'].mean())

# Convert 'Year' to integer and filter data for the last 6 years (2018-2023)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[df['Year'].between(2018, 2023)]

# Remove rows with missing values in critical columns (optional)
df.dropna(subset=['GIS Calculated Acres', 'Cause_Category'], inplace=True)

# One-hot encoding for 'Cause' and 'Cause_Category' columns
df_encoded = pd.get_dummies(df, columns=['Cause', 'Cause_Category'], drop_first=True)

# Dependent Variable: binarizing fire size based on a threshold
threshold = df['GIS Calculated Acres'].mean()  # Set a threshold to classify fires into 'large' or 'small'
df['Fire_Size_Category'] = df['GIS Calculated Acres'].apply(lambda x: 1 if x > threshold else 0)

# Feature Selection: using the one-hot encoded 'Cause' and other features as independent variables
X = df_encoded.drop(['GIS Calculated Acres', 'Fire_Size_Category'], axis=1, errors='ignore')
y = df['Fire_Size_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# Logistic Regression Model (Dropping NaN values)
# ========================
X_train_logistic = X_train.dropna()
y_train_logistic = y_train[X_train_logistic.index]  # Ensure y is aligned with the reduced X after dropping NaN rows
X_test_logistic = X_test.dropna()
y_test_logistic = y_test[X_test_logistic.index]

# Initialize and train the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_logistic, y_train_logistic)

# Predictions for Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_logistic)

# Evaluate Logistic Regression Model
print("Logistic Regression Model")
print(f"Accuracy: {accuracy_score(y_test_logistic, y_pred_logistic) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_logistic, y_pred_logistic))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_logistic, y_pred_logistic))

# ========================
# Gradient Boosting Model (Retaining NaN values)
# ========================
gradient_model = HistGradientBoostingClassifier()
gradient_model.fit(X_train, y_train)

# Predictions for Gradient Boosting
y_pred_gradient = gradient_model.predict(X_test)

# Evaluate Gradient Boosting Model
print("\nGradient Boosting Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gradient) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gradient))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_gradient))
