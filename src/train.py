import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json

# Load the dataset
# You'll need to have the titanic.csv file in the same directory or specify the correct path
titanic = pd.read_csv('data/titanic.csv')

# Data Cleaning and Preparation (basic example)
# Handle missing values (e.g., fill 'Age' with the mean, drop 'Cabin')
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic.drop('Cabin', axis=1, inplace=True)

# Convert categorical variables into numerical ones using one-hot encoding
titanic = pd.get_dummies(titanic, columns=['Sex', 'Embarked'], drop_first=True)

# Select features (X) and target variable (y)
X = titanic[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = titanic['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1}")

# You can also use other evaluation metrics like confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and metrics
joblib.dump(model, 'models/titanic_logistic_regression_model.pkl')
with open('metrics_logistic_regression.json', 'w') as f:
    json.dump({'accuracy': accuracy, 'f1_score': f1}, f)

print(f"Model trained. Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}")
