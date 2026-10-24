import json
import pandas as pd

# 1. Ingest JSON files
with open("acndata_sessions (1).json", "r") as f:
    d1 = json.load(f)["_items"]
with open("acndata_sessions.json", "r") as f:
    d2 = json.load(f)["_items"]

df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

# 2. Parse timestamps and energy
date_columns = ["connectionTime", "disconnectTime", "doneChargingTime"]
for df in [df1, df2]:
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")

required_columns = ["sessionID", "stationID", "connectionTime", "disconnectTime", "kWhDelivered"]
clean1 = df1.dropna(subset=required_columns).copy()
clean2 = df2.dropna(subset=required_columns).copy()

clean1 = clean1[(clean1["kWhDelivered"] >= 0.5) & (clean1["disconnectTime"] > clean1["connectionTime"])].copy()
clean2 = clean2[(clean2["kWhDelivered"] >= 0.5) & (clean2["disconnectTime"] > clean2["connectionTime"])].copy()

# 3. Calculate durations and fix missing values
for df in [clean1, clean2]:
    invalid_done = (
        df["doneChargingTime"].isna()
        | (df["doneChargingTime"] > df["disconnectTime"])
        | (df["doneChargingTime"] < df["connectionTime"])
    )
    df.loc[invalid_done, "doneChargingTime"] = df.loc[invalid_done, "disconnectTime"]
    
    df["session_duration_minutes"] = (df["disconnectTime"] - df["connectionTime"]).dt.total_seconds() / 60.0
    df["active_charging_minutes"] = (df["doneChargingTime"] - df["connectionTime"]).dt.total_seconds() / 60.0
    df["active_charging_minutes"] = df["active_charging_minutes"].clip(lower=3.0)
    df["idle_duration_minutes"] = (df["disconnectTime"] - df["doneChargingTime"]).dt.total_seconds() / 60.0
    df["charging_rate_kw"] = df["kWhDelivered"] / (df["active_charging_minutes"] / 60.0)

    local_time = df["connectionTime"].dt.tz_convert("America/Los_Angeles")
    df["date"] = local_time.dt.date
    df["hour"] = local_time.dt.hour
    df["day_of_week"] = local_time.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# 4. Filter physical charger caps & multi-day outliers
clean1 = clean1[(clean1["session_duration_minutes"] <= 24 * 60) & (clean1["charging_rate_kw"] <= 25.0)].copy()
clean2 = clean2[(clean2["session_duration_minutes"] <= 24 * 60) & (clean2["charging_rate_kw"] <= 25.0)].copy()

# 5. Merge, sort, and standardize
merged_df = pd.concat([clean1, clean2], ignore_index=True)
merged_df = merged_df.sort_values("connectionTime").reset_index(drop=True)

for col in ["session_duration_minutes", "active_charging_minutes", "kWhDelivered"]:
    mean_val = merged_df[col].mean()
    std_val = merged_df[col].std()
    merged_df[col + "_zscore"] = (merged_df[col] - mean_val) / std_val
    merged_df[col + "_outlier"] = merged_df[col + "_zscore"].abs() > 3

# 6. Export clean tabular CSV
merged_df.to_csv("caltech_acn_cleaned_merged.csv", index=False)
print(f"Dataset ready for Phase 3! Shape: {merged_df.shape}")