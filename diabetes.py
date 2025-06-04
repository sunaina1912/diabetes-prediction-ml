"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load dataset
data = pd.read_csv('diabetes.csv')

# Display the first few rows
print(data.head())

# Basic dataset information
print(data.info())
print(data.describe())
# Check for missing values
print(data.isnull().sum())

# Check for any abnormal values (like 0 in columns where 0 isn't possible, e.g., 'Glucose', 'BloodPressure')
print((data == 0).sum())

# Replace zeroes with NaN for 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI'
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, np.nan)

# Impute missing values with the median
data.fillna(data.median(), inplace=True)

# Normalize feature columns
scaler = StandardScaler()

data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
# Visualize correlation
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribution of classes
sns.countplot(x='Outcome', data=data)
plt.title("Distribution of Outcome Variable")
plt.show()
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


import joblib

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
print("Model saved as 'diabetes_model.pkl'")
joblib.dump(scaler, 'scalar.pkl')


print("Model saved as 'scalar.pkl'")""""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# Load the dataset (Pima Indians Diabetes Database)
data = pd.read_csv('diabetes.csv')

# Handle missing values or zeros in certain columns if any
data = data.replace(0, np.nan).fillna(data.mean())

# Split the data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
