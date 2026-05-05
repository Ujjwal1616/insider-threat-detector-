import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import config

url = URL.create(
    drivername="mysql+pymysql",
    username=config.DB_USER,
    password=config.DB_PASSWORD,
    host=config.DB_HOST,
    port=config.DB_PORT,
    database=config.DB_NAME,
)
engine = create_engine(url)

df = pd.read_csv("data/final_with_insider.csv")
df = df.rename(columns={"user": "user_id", "day": "session_date"})
df = df.drop(columns=["downloads", "failed_login", "insider"], errors="ignore")

# Fix date format from DD-MM-YYYY to YYYY-MM-DD for MySQL
df["session_date"] = pd.to_datetime(df["session_date"], dayfirst=True).dt.strftime("%Y-%m-%d")

if "is_weekend" not in df.columns:
    df["is_weekend"] = 0

users = df["user_id"].unique()
df = df[df["user_id"].isin(users)]
df.to_sql("user_sessions", engine, if_exists="append", index=False, chunksize=500)
print(f"Seeded {len(df)} rows for {len(users)} users successfully")