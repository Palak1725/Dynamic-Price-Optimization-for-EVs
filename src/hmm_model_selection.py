import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ---------- Job 1: Build the observation matrix X ----------
df = pd.read_csv("ev_charging_hourly_features.csv")
X = df[["occupancy_rate", "transformer_load_ratio"]].values

print("Shape of X:", X.shape)          # should be (8760, 2)
print(df[["occupancy_rate", "transformer_load_ratio"]].describe())
print()

# ---------- Job 2: Compare K = 2, 3, 4, 5 using AIC and BIC ----------
T, D = X.shape
rows = []

for K in [2, 3, 4, 5]:
    model = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=200, random_state=42)
    model.fit(X)

    log_l = model.score(X)             # this is ln(L)

    # Count of free parameters (my own count)
    p = (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2
    # The formula written on your team sheet
    p_sheet = K**2 + 4 * K - 1

    aic = -2 * log_l + 2 * p
    bic = -2 * log_l + p * np.log(T)

    rows.append({"K": K, "logL": round(log_l, 2), "p": p,
                 "p_sheet": p_sheet, "AIC": round(aic, 2),
                 "BIC": round(bic, 2)})

results = pd.DataFrame(rows)
print(results.to_string(index=False))
print()
print("Best K by AIC:", results.loc[results["AIC"].idxmin(), "K"])
print("Best K by BIC:", results.loc[results["BIC"].idxmin(), "K"])
results.to_csv("aic_bic_results.csv", index=False)