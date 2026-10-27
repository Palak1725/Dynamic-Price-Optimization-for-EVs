import os
import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM

# ---------- Build X ----------
df = pd.read_csv("ev_charging_hourly_features.csv")
X = df[["occupancy_rate", "transformer_load_ratio"]].values

# ---------- Train the production model (K = 3) ----------
model = GaussianHMM(n_components=3, covariance_type="full",
                    n_iter=200, random_state=42)
model.fit(X)

# ---------- Extract parameters ----------
print("Transition matrix (3x3):")
print(np.round(model.transmat_, 3))
print("\nEmission means [occupancy, load_ratio]:")
print(np.round(model.means_, 4))
print("\nCovariance matrices:")
print(np.round(model.covars_, 5))
print("\nStarting probabilities:", np.round(model.startprob_, 3))

# ---------- Decode states (Viterbi) ----------
raw_states = model.predict(X)

# State numbers are random, so sort them by average transformer load:
# lowest load = Surplus, middle = Nominal, highest = Peak Stress
order = np.argsort(model.means_[:, 1])
rank = np.empty_like(order)
rank[order] = np.arange(3)
named_states = rank[raw_states]   # 0 = Surplus, 1 = Nominal, 2 = Peak
labels = {0: "Surplus", 1: "Nominal", 2: "Peak Stress"}

print("\nHours spent in each state:")
for k in range(3):
    print(f"  {labels[k]}: {(named_states == k).sum()} hours")

print("\nPersistence (diagonal of transition matrix, should be > 0.65):")
for k in range(3):
    print(f"  {labels[k]}: {model.transmat_[order[k], order[k]]:.3f}")

# ---------- Save outputs ----------
os.makedirs("trained_models", exist_ok=True)
joblib.dump(model, "trained_models/gaussian_hmm_v1.pkl")

out = df.copy()
out["raw_state"] = raw_states
out["state"] = named_states
out["state_name"] = out["state"].map(labels)
out.to_csv("hourly_features_with_states.csv", index=False)
print("\nSaved model and states file.")