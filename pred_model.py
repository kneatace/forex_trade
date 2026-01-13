import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

#load and clean data
df = pd.read_csv("/home/nitesh/Documents/de_task/forex_usd_npr.csv", parse_dates=["rate_date"])
df = df.sort_values("rate_date").reset_index(drop=True)

#ensure numeric
df['buying_rate'] = pd.to_numeric(df['buying_rate'], errors='coerce')
df = df[df['buying_rate'].notna()]
df = df[df['buying_rate'] > 0]

#handle missing days like weekends/holidays
df = df.set_index('rate_date').asfreq('D')
df['buying_rate'] = df['buying_rate'].ffill()
df = df.reset_index()

#lag features of last 10 days
for lag in range(1, 11):
    df[f'lag_{lag}'] = df['buying_rate'].shift(lag)

#exponential moving averages
df['ema_5'] = df['buying_rate'].ewm(span=5, adjust=False).mean()
df['ema_14'] = df['buying_rate'].ewm(span=14, adjust=False).mean()

#rate of change
df['roc_1'] = df['buying_rate'].pct_change()

#volatility (rolling std)
df['std_7'] = df['buying_rate'].rolling(7).std()

#target: 1 if next day's rate > today, else 0
df['target'] = (df['buying_rate'].shift(-1) > df['buying_rate']).astype(int)

#drop rows with NaN due to rolling and lag features
df = df.dropna().reset_index(drop=True)

#feature columns
feature_cols = [f'lag_{i}' for i in range(1, 11)] + ['ema_5','ema_14','roc_1','std_7']

#Time-Series Cross-Validation
X = df[feature_cols]
y = df['target']

tscv = TimeSeriesSplit(n_splits=5)
cv_accuracies, cv_precisions, cv_recalls = [], [], []

for train_idx, test_idx in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

    #scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cv)
    X_test_scaled = scaler.transform(X_test_cv)

    #train RF
    rf_cv = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_cv.fit(X_train_scaled, y_train_cv)

    #predict
    y_pred_cv = rf_cv.predict(X_test_scaled)

    #evaluate
    cv_accuracies.append(accuracy_score(y_test_cv, y_pred_cv))
    cv_precisions.append(precision_score(y_test_cv, y_pred_cv))
    cv_recalls.append(recall_score(y_test_cv, y_pred_cv))

print("Time-Series CV Metrics:")
print(f"Accuracy:  {np.mean(cv_accuracies):.4f}")
print(f"Precision: {np.mean(cv_precisions):.4f}")
print(f"Recall:    {np.mean(cv_recalls):.4f}")

#final training & validation of last 90 Days)
train = df[:-90]
test = df[-90:]

X_train_final = train[feature_cols]
y_train_final = train['target']
X_test_final = test[feature_cols]
y_test_final = test['target']

#scale
scaler_final = StandardScaler()
X_train_scaled_final = scaler_final.fit_transform(X_train_final)
X_test_scaled_final = scaler_final.transform(X_test_final)

#train final RF
rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
rf_final.fit(X_train_scaled_final, y_train_final)

#predict last 90 days
y_pred_final = rf_final.predict(X_test_scaled_final)

#evaluate
accuracy_final = accuracy_score(y_test_final, y_pred_final)
precision_final = precision_score(y_test_final, y_pred_final)
recall_final = recall_score(y_test_final, y_pred_final)

print("\nRandom Forest Performance on Last 90 Days:")
print(f"Accuracy:  {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall:    {recall_final:.4f}")

#export results
results = test[['rate_date','buying_rate']].copy()
results['predicted_up'] = y_pred_final
results['actual_up'] = y_test_final.values

excel_file = "forex_prediction_results.xlsx"
results.to_excel(excel_file, index=False)
print(f"\nResults exported to {excel_file} successfully!")

#for predicting tomorrow's rate using actual latest data of 10 days
last_10_days = df['buying_rate'].iloc[-10:].values[::-1]  # latest to oldest
lag_features = {f'lag_{i+1}': last_10_days[i] for i in range(10)}

#EMA, ROC, Volatility
ema_5 = df['buying_rate'].ewm(span=5, adjust=False).mean().iloc[-1]
ema_14 = df['buying_rate'].ewm(span=14, adjust=False).mean().iloc[-1]
roc_1 = df['buying_rate'].pct_change().iloc[-1]
std_7 = df['buying_rate'].rolling(7).std().iloc[-1]

# Prepare DataFrame
tomorrow_features = pd.DataFrame([{
    **lag_features,
    'ema_5': ema_5,
    'ema_14': ema_14,
    'roc_1': roc_1,
    'std_7': std_7
}])

# Scale
tomorrow_scaled = scaler_final.transform(tomorrow_features)

# Predict
tomorrow_pred = rf_final.predict(tomorrow_scaled)[0]
print(f"\nPrediction for tomorrow: {'Increase' if tomorrow_pred == 1 else 'Decrease'}")
