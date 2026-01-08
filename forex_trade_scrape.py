import requests
import mysql.connector
from datetime import datetime, timedelta, date
import time

# Mysql connection setup
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",    
    database="forex_trade"
)
cursor = db.cursor()

# Create table if not exists
cursor.execute("""
create table if not exists forex_rate (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rate_date DATE NOT NULL,
    currency_code CHAR(3) NOT NULL,
    currency_name VARCHAR(50),
    unit INT NOT NULL,
    buying_rate DECIMAL(10,4),
    selling_rate DECIMAL(10,4),
    source VARCHAR(20) DEFAULT 'NRB',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_forex (rate_date, currency_code)
)
""")
db.commit()

# NRB Forex API settings
API_URL = "https://www.nrb.org.np/api/forex/v1/rates"
PER_PAGE = 100
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime.today()

# Function to insert data
def insert_forex_rate(rate_date, currency_code, currency_name, unit, buying_rate, selling_rate):
    sql = """
    insert into forex_rate
    (rate_date, currency_code, currency_name, unit, buying_rate, selling_rate)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        buying_rate = VALUES(buying_rate),
        selling_rate = VALUES(selling_rate)
    """
    cursor.execute(sql, (rate_date, currency_code, currency_name, unit, buying_rate, selling_rate))

# Loop through years
current_start = START_DATE
while current_start <= END_DATE:
    # Fetch one year at a time
    current_end = min(current_start + timedelta(days=364), END_DATE)
    page = 1
    
    while True:
        payload = {
            "from": current_start.strftime("%Y-%m-%d"),
            "to": current_end.strftime("%Y-%m-%d"),
            "per_page": PER_PAGE,
            "page": page
        }

        try:
            response = requests.get(API_URL, params=payload)
            response.raise_for_status()
            data = response.json()

            # Access the nested payload
            payloads = data.get("data", {}).get("payload", [])
            if not payloads:
                break

            for day in payloads:
                rate_date = day.get("date")
                rates = day.get("rates", [])
                for rate in rates:
                    currency = rate.get("currency", {})
                    insert_forex_rate(
                        rate_date=rate_date,
                        currency_code=currency.get("iso3"),
                        currency_name=currency.get("name"),
                        unit=currency.get("unit", 1),
                        buying_rate=float(rate.get("buy") or 0),
                        selling_rate=float(rate.get("sell") or 0)
                    )

            db.commit()
            print(f"Inserted page {page} for {current_start.date()} → {current_end.date()} with {len(payloads)} payloads.")
            page += 1
            time.sleep(0.3)  # avoid api rate limits

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            break

    # Move to next year
    current_start = current_end + timedelta(days=1)

# Close DB connection
cursor.close()
db.close()
print("Extraction complete and DB connection closed.")