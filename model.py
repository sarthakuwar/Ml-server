import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump

# Load the dataset (using the dummy data example)
data = pd.DataFrame({
    'past_sales': np.random.randint(0, 100, size=100),
    'is_festival': np.random.choice([0, 1], size=100),
    'discount': np.random.choice([0, 5, 10, 15, 20], size=100),
    'availability_nearby': np.random.randint(0, 50, size=100),
    'quantity_ordered': np.random.randint(0, 150, size=100)
})

# Select features and target
X = data[['past_sales', 'is_festival', 'discount', 'availability_nearby']]
y = data['quantity_ordered']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model using joblib
dump(model, 'demand_model.joblib')
