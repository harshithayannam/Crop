import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('crop_production.csv')

# Preprocess the data
data.dropna(inplace=True)

# Select features and target
X = data[['Area']].values
y = data['Production'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict function
def predict_crop_yield(area):
    return model.predict(np.array([[area]]))[0]

# Example usage
area = 500
predicted_yield = predict_crop_yield(area)
print(f'Predicted crop yield for an area of {area}: {predicted_yield}')
