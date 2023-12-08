import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
data = pd.read_csv('/content/data.csv')
x= data.drop('price', axis=1)
y = data['price']
numeric_columns = x.select_dtypes(include=['number']).columns
x_numeric = x[numeric_columns]
x_train, x_test, y_train, y_test = train_test_split(x_numeric, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[x_train_scaled.shape[1]]),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
print("Input Shape:", x_train_scaled.shape)
model.fit(x_train_scaled, y_train, epochs=50, verbose=0)
y_pred_nn = model.predict(x_test_scaled).flatten()
linear_model = LinearRegression()
linear_model.fit(x_trained, y_train)
y_pred_linear = linear_model.predict(x_tested)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("Linear Regression Metrics:")
print(f'Mean Absolute Error: {mae_linear}')
print(f'Mean Squared Error: {mse_linear}')
print(f'R-squared: {r2_linear}')
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print("\nNeural Network Metrics:")
print(f'Mean Absolute Error: {mae_nn}')
print(f'Mean Squared Error: {mse_nn}')
print(f'R-squared: {r2_nn}')
input_data = {
    'bedrooms': 3,
    'bathrooms':2,
    'sqft_living':1500,
    'sqft_lot':2000,
    'floors':1.5,
     'waterfront':0,
    'view':0,
    'condition':5,
     'sqft_above': 2000,
    'sqft_basement':0,
    'yr_built':1957,
    'yr_renovated':0
}
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
predicted_price = model.predict(input_scaled)
print(f'Predicted House Price: ${predicted_price[0]}')
