import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import numpy as np

# Load the dataset from the CSV file
data = pd.read_csv('dummy.csv')

# Print the columns to debug
print("Columns in the dataset:", data.columns)

# Select features and target
# Adjust the feature selection based on available columns
X = data[['StockQuantity', 'LowStockWarning', 'PurchasePrice', 'InclusiveOfTax']]
y = data['StockQuantity']  # Assuming 'StockQuantity' is the target for demonstration

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model using joblib
dump(model, 'demand_model.joblib')

# Save the results to a CSV file
results = pd.DataFrame({
    'actual': y_test,
    'predicted': model.predict(X_test)
})
results.to_csv('predictions.csv', index=False)