import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('D:\\bankloans.csv')
print(data.head())

# Handle missing values
data = data.fillna(data.mean())

print(data.info())

# Summary statistics
print(data.describe())

# Distribution of features
data.hist(bins=30, figsize=(15, 10))
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Split data into train and test sets
X = data.drop('default', axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Choose a regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# Plot actual vs. predicted risk scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Risk Scores')
plt.ylabel('Predicted Risk Scores')
plt.title('Actual vs. Predicted Risk Scores')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.show()

# Save the model
joblib.dump(model, 'risk_model.pkl')

    # Load the model
model = joblib.load('risk_model.pkl')

# Now you can use the loaded model
print(model)
