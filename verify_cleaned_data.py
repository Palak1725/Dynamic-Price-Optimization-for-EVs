import pandas as pd

df = pd.read_csv("caltech_acn_cleaned_merged.csv")
print("Dataset shape:", df.shape)

# Check duplicates and missing values
print("Duplicate sessions:", df["sessionID"].duplicated().sum())

cols_to_check = [
    "sessionID",
    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "kWhDelivered",
    "session_duration_minutes",
    "active_charging_minutes",
    "idle_duration_minutes",
    "charging_rate_kw",
]
print("\nMissing values:")
print(df[cols_to_check].isna().sum())

# Timestamp sequence and ordering
conn = pd.to_datetime(df["connectionTime"])
disc = pd.to_datetime(df["disconnectTime"])
done = pd.to_datetime(df["doneChargingTime"])

print("\nTimestamp checks:")
print("Illogical disconnects (conn >= disc):", (conn >= disc).sum())
print("Illogical charge end (done > disc):", (done > disc).sum())
print("Chronological sort check:", conn.is_monotonic_increasing)

# Duration and power bounds
print("\nBoundary checks:")
print("Sessions <= 0 mins:", (df["session_duration_minutes"] <= 0).sum())
print("Sessions > 24 hrs:", (df["session_duration_minutes"] > 1440).sum())
print("Negative idle mins:", (df["idle_duration_minutes"] < 0).sum())
print("Max charging rate (kW):", round(df["charging_rate_kw"].max(), 2))
print("Min energy (kWh):", df["kWhDelivered"].min())

# Summary stats
print("\nSummary stats:")
print(
    df[
        [
            "session_duration_minutes",
            "active_charging_minutes",
            "idle_duration_minutes",
            "kWhDelivered",
        ]
    ]
    .describe()
    .round(2)
)