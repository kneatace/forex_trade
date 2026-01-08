import mysql.connector
import pandas as pd

# -----------------------------
# 1️⃣ MySQL connection setup
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",        # your MySQL password
    database="forex_trade"
)

# -----------------------------
# 2️⃣ Query the table
# -----------------------------
query = "SELECT * FROM forex_rate WHERE currency_code='USD';"

df = pd.read_sql(query, con=db)

# -----------------------------
# 3️⃣ Export to CSV
# -----------------------------
csv_file = "forex_usd_npr.csv"
df.to_csv(csv_file, index=False)
print(f"Data exported to {csv_file} successfully!")

# -----------------------------
# 4️⃣ Close DB connection
# -----------------------------
db.close()
