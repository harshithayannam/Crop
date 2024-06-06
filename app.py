from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

# Load the dataset
data = pd.read_csv('crop_production.csv')
data.dropna(inplace=True)
X = data[['Area']].values
y = data['Production'].values

# Train the regression model
model = LinearRegression()
model.fit(X, y)

def predict_crop_yield(area):
    return model.predict(np.array([[area]]))[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        area = float(request.form['area'])
        prediction = predict_crop_yield(area)
        return render_template('index.html', prediction_text=f'Predicted Crop Yield: {prediction:.2f}')
    
if __name__ == "__main__":
    app.run(debug=True)
