import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

#Load & clean data
df = pd.read_csv("/home/nitesh/Documents/de_task/forex_usd_npr.csv", parse_dates=["rate_date"])
df = df.sort_values("rate_date").reset_index(drop=True)

# Ensure numeric
df['buying_rate'] = pd.to_numeric(df['buying_rate'], errors='coerce')
df = df[df['buying_rate'].notna()]
df = df[df['buying_rate'] > 0]

#data cleaning
df = df.set_index('rate_date').asfreq('D')
df['buying_rate'] = df['buying_rate'].ffill()
df = df.reset_index()

# Lag features: last 10 days
for lag in range(1, 11):
    df[f'lag_{lag}'] = df['buying_rate'].shift(lag)

# Exponential moving averages
df['ema_5'] = df['buying_rate'].ewm(span=5, adjust=False).mean()
df['ema_14'] = df['buying_rate'].ewm(span=14, adjust=False).mean()

# Rate of change
df['roc_1'] = df['buying_rate'].pct_change()

# Volatility (rolling std dev)
df['std_7'] = df['buying_rate'].rolling(7).std()

# Target: 1 if next day's rate > today, else 0
df['target'] = (df['buying_rate'].shift(-1) > df['buying_rate']).astype(int)

# Drop rows with NaN from lags/rolling
df = df.dropna().reset_index(drop=True)

# Feature columns
feature_cols = [f'lag_{i}' for i in range(1, 11)] + ['ema_5','ema_14','roc_1','std_7']

X = df[feature_cols]
y = df['target']

#Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

cv_accuracies = []
cv_precisions = []
cv_recalls = []

for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cv)
    X_test_scaled = scaler.transform(X_test_cv)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train_cv)
    
    # Predict
    y_pred_cv = rf.predict(X_test_scaled)
    
    # Evaluate
    cv_accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
    cv_precisions.append(precision_score(y_test_cv, y_pred_cv))
    cv_recalls.append(recall_score(y_test_cv, y_pred_cv))

print("Time-Series CV Metrics:")
print(f"Accuracy:  {np.mean(cv_accuracies):.4f}")
print(f"Precision: {np.mean(cv_precisions):.4f}")
print(f"Recall:    {np.mean(cv_recalls):.4f}")

#Final training & prediction for last 90 days
final_train = df[:-90]
final_test = df[-90:]

X_train_final = final_train[feature_cols]
y_train_final = final_train['target']
X_test_final = final_test[feature_cols]
y_test_final = final_test['target']

# Scale
scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_final)
X_test_scaled_final = scaler_final.transform(X_test_final)

# Train RF
rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
rf_final.fit(X_train_scaled_final, y_train_final)

# Predict
y_pred_final = rf_final.predict(X_test_scaled_final)

# Evaluation
accuracy_final = accuracy_score(y_test_final, y_pred_final)
precision_final = precision_score(y_test_final, y_pred_final)
recall_final = recall_score(y_test_final, y_pred_final)

print("\nRandom Forest Performance on Last 90 Days:")
print(f"Accuracy:  {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")

#Compare & export predictions

results = final_test[['rate_date','buying_rate']].copy()
results['predicted_up'] = y_pred_final
results['actual_up'] = y_test_final.values

excel_file = "forex_prediction_results.xlsx"
results.to_excel(excel_file, index=False)
print(f"\nResults exported to {excel_file} successfully!")

# Prepare features for tomorrow prediction
last_10_days = df['buying_rate'].iloc[-10:].values[::-1]  # latest to oldest
lag_features = {f'lag_{i+1}': last_10_days[i] for i in range(10)}

#EMA
ema_5 = df['buying_rate'].ewm(span=5, adjust=False).mean().iloc[-1]
ema_14 = df['buying_rate'].ewm(span=14, adjust=False).mean().iloc[-1]

# Rate of change and volatility
roc_1 = df['buying_rate'].pct_change().iloc[-1]
std_7 = df['buying_rate'].rolling(7).std().iloc[-1]

# Create DataFrame
tomorrow_features = pd.DataFrame([{
    **lag_features,
    'ema_5': ema_5,
    'ema_14': ema_14,
    'roc_1': roc_1,
    'std_7': std_7
}])

# Scale
tomorrow_scaled = scaler.transform(tomorrow_features)

# Predict
tomorrow_pred = rf.predict(tomorrow_scaled)[0]
print(f"Prediction for tomorrow: {'Increase' if tomorrow_pred == 1 else 'Decrease'}")

