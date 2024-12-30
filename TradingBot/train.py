import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from DataFetching.Database import gen,ex

data_1 = list(gen.find())
data_2 = list(ex.find())

df1 = pd.DataFrame(data_1)
df2 = pd.DataFrame(data_2)

df1['timestamp'] = pd.to_datetime(df1['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
df2['timestamp'] = pd.to_datetime(df2['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

# Merge the datasets on Timestamp
merged_df = pd.merge(df1, df2, on='timestamp', how='inner')
merged_df.drop(['_id_x', '_id_y'])

merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])

# Add features (e.g., gas fee, transaction count, volume, etc.)
merged_df['Price Diff'] = merged_df['Close Price'] - merged_df['Open Price']

# Target: Predict the next close price
merged_df['Next Close'] = merged_df['Close Price'].shift(-1)
merged_df = merged_df.dropna()  # Remove rows with missing Next Close

# Features and target
features = ['Gas Fee (Gwei)', 'Transaction Count', 'Volume', 'Price Diff']
target = 'Next Close'

X = merged_df[features]
y = merged_df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a regression model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Example prediction
print("Predicted next prices:", predictions[:5])