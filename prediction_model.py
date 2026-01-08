import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1️⃣ Load Forex Data
# -----------------------------
df = pd.read_csv("/home/nitesh/Documents/de_task/forex_usd_npr.csv", parse_dates=["rate_date"])
df = df.sort_values("rate_date").reset_index(drop=True)

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
# Create lag features (last 10 days)
for lag in range(1, 11):
    df[f'lag_{lag}'] = df['buying_rate'].shift(lag)

# Exponential Moving Averages
df['ema_5'] = df['buying_rate'].ewm(span=5, adjust=False).mean()
df['ema_14'] = df['buying_rate'].ewm(span=14, adjust=False).mean()

# Rate of change
df['roc_1'] = df['buying_rate'].pct_change()

# Volatility (std dev)
df['std_7'] = df['buying_rate'].rolling(7).std()

# Target: 1 if next day's rate > today, else 0
df['target'] = (df['buying_rate'].shift(-1) > df['buying_rate']).astype(int)

# Drop rows with NaN (due to rolling and lag features)
df = df.dropna().reset_index(drop=True)

# -----------------------------
# 3️⃣ Train/Test Split
# -----------------------------
# Last 90 days are validation
train = df[:-90]
test = df[-90:]

feature_cols = [f'lag_{i}' for i in range(1, 6)] + ['ema_5','ema_14','roc_1','std_7']

X_train = train[feature_cols]
y_train = train['target']

X_test = test[feature_cols]
y_test = test['target']

# Optional: scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4️⃣ Model Training (Random Forest)
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# -----------------------------
# 5️⃣ Prediction
# -----------------------------
y_pred = rf.predict(X_test_scaled)

# -----------------------------
# 6️⃣ Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Random Forest Performance on Last 90 Days:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# -----------------------------
# 7️⃣ Compare Predictions with Actual
# -----------------------------
results = test[['rate_date','buying_rate']].copy()
results['predicted_up'] = y_pred
results['actual_up'] = y_test.values

print("\nPredicted vs Actual (last 90 days):")
print(results)
excel_file = "forex_prediction_results.xlsx"
results.to_excel(excel_file, index=False)
print(f"\n Results exported to {excel_file} successfully!")
