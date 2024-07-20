# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset 
data = pd.read_csv('HousePrediction.csv')

# Assume 'price' is our target variable, and 'size', 'bedrooms', 'bathrooms' are features
X = data[['size', 'bedrooms', 'bathrooms']].values
y = data['price'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting predictions vs actual
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Example of predicting a new house price
new_house = np.array([[2500, 4, 3]])  # Size: 2500 sqft, Bedrooms: 4, Bathrooms: 3
predicted_price = model.predict(new_house)
print(f'Predicted price for the new house: ${predicted_price[0]:,.2f}')
