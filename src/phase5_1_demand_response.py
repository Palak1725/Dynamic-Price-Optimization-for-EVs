
"""
PHASE 5.1: STATE-DEPENDENT PRICE ELASTICITY
EV Charging Demand Response Prediction
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "hourly_features_with_states.csv"

MODEL_FILE = (
    ROOT / "trained_models" / "gaussian_hmm_ordered_v1.pkl"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phase5_1_demand_response.csv"
SUMMARY_FILE = OUTPUT_DIR / "phase5_1_scenario_summary.csv"


# ==========================================
# 2. CONFIGURATION
# ==========================================

TRANSFORMER_CAPACITY_KW = 150.0

STATE_NAMES = {
    0: "Surplus",
    1: "Nominal",
    2: "Peak Stress"
}

# Assumed elasticity values, not experimentally fitted.
ELASTICITY = {
    0: -1.2,
    1: -0.8,
    2: -0.4
}

FEATURES = [
    "occupancy_rate",
    "transformer_load_ratio"
]

PRICE_SCENARIOS = {
    "price_down_10pct": 0.90,
    "price_baseline": 1.00,
    "price_up_10pct": 1.10
}


# ==========================================
# 3. LOAD DATA AND ORDERED HMM
# ==========================================

print("PHASE 5.1: DEMAND RESPONSE PREDICTION")

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)

required_columns = FEATURES + [
    "timestamp",
    "base_tou_price"
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

X = df[FEATURES].to_numpy(dtype=float)

if not np.isfinite(X).all():
    raise ValueError("HMM features contain invalid values.")

if (df["base_tou_price"] <= 0).any():
    raise ValueError("Baseline prices must be positive.")

if (df["transformer_load_ratio"] < 0).any():
    raise ValueError("Negative transformer load detected.")

print("Dataset loaded:", df.shape)
print("Ordered HMM model loaded.")


# ==========================================
# 4. CALCULATE SOFT STATE PROBABILITIES
# ==========================================

# Each row contains the probability of belonging
# to Surplus, Nominal and Peak Stress respectively.

state_probabilities = model.predict_proba(X)

if state_probabilities.shape != (len(df), 3):
    raise ValueError("Unexpected HMM probability dimensions.")

if not np.allclose(
    state_probabilities.sum(axis=1), 1.0, atol=1e-6
):
    raise ValueError("State probabilities do not sum to 1.")

df["prob_surplus"] = state_probabilities[:, 0]
df["prob_nominal"] = state_probabilities[:, 1]
df["prob_peak_stress"] = state_probabilities[:, 2]

print("Soft state probabilities calculated.")


# ==========================================
# 5. BASELINE CHARGING DEMAND
# ==========================================

# Following the Phase 4.4 convention:
# Hourly demand = transformer load ratio * 150 kW * 1 hour.

df["baseline_demand_kwh"] = (
    df["transformer_load_ratio"]
    * TRANSFORMER_CAPACITY_KW
)

df["baseline_price"] = df["base_tou_price"]


# ==========================================
# 6. STATE-DEPENDENT DEMAND PREDICTION
# ==========================================

# For each price scenario, calculate the predicted
# demand under each possible operating state.
#
# Then combine state-specific demand using the
# HMM posterior probabilities.

for scenario, price_factor in PRICE_SCENARIOS.items():

    proposed_price = (
        df["baseline_price"] * price_factor
    )

    state_demands = np.zeros((len(df), 3))

    for state in range(3):

        epsilon = ELASTICITY[state]

        state_demands[:, state] = (
            df["baseline_demand_kwh"].to_numpy()
            * (
                proposed_price.to_numpy()
                / df["baseline_price"].to_numpy()
            ) ** epsilon
        )

    # Posterior-weighted expected demand
    df[f"predicted_{scenario}_kwh"] = (
        state_probabilities * state_demands
    ).sum(axis=1)


# ==========================================
# 7. EXPECTED ELASTICITY (DIAGNOSTIC)
# ==========================================

elasticity_values = np.array([
    ELASTICITY[0],
    ELASTICITY[1],
    ELASTICITY[2]
])

df["posterior_weighted_elasticity"] = (
    state_probabilities * elasticity_values
).sum(axis=1)


# ==========================================
# 8. DEMAND CHANGE CALCULATION
# ==========================================

df["demand_change_up_10pct_kwh"] = (
    df["predicted_price_up_10pct_kwh"]
    - df["baseline_demand_kwh"]
)

df["demand_change_down_10pct_kwh"] = (
    df["predicted_price_down_10pct_kwh"]
    - df["baseline_demand_kwh"]
)


# ==========================================
# 9. SAVE HOURLY RESULTS
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# 10. CREATE SCENARIO SUMMARY
# ==========================================

summary_rows = []

for scenario, factor in PRICE_SCENARIOS.items():

    predicted = df[f"predicted_{scenario}_kwh"]

    baseline = df["baseline_demand_kwh"].sum()

    total_predicted = predicted.sum()

    change_pct = (
        (total_predicted - baseline) / baseline * 100
        if baseline > 0 else np.nan
    )

    summary_rows.append({
        "scenario": scenario,
        "price_change_pct": (factor - 1) * 100,
        "baseline_demand_kwh": baseline,
        "predicted_demand_kwh": total_predicted,
        "demand_change_pct": change_pct
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(SUMMARY_FILE, index=False)


# ==========================================
# 11. FINAL REPORT
# ==========================================

print("\n========== PHASE 5.1 RESULTS ==========")

print(summary.to_string(index=False))

print("\nState Probability Averages:")

print(
    df[
        [
            "prob_surplus",
            "prob_nominal",
            "prob_peak_stress"
        ]
    ].mean()
)

print("\nFiles saved successfully:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nPHASE 5.1 EXECUTION COMPLETED.")
