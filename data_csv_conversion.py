import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd

#Mysql connection setup
load_dotenv()
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
#Query the table
query = "select * from forex_rate where currency_code='USD';"

df = pd.read_sql(query, con=db)

#Export to CSV
csv_file = "forex_usd_npr.csv"
df.to_csv(csv_file, index=False)
print(f"Data exported to {csv_file} successfully!")

#Close database connection
db.close()
