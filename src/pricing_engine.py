#phase 4.4
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 1. Load the certified Phase 4.3 dataset
csv_path = "hourly_features_with_states.csv"
df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 65)
print("PHASE 4.4: DYNAMIC PRICING OPTIMIZATION ENGINE")
print("=" * 65)
print(f"Loaded {len(df)} continuous hours from '{csv_path}'")

# 2. Compute physical active energy in kWh (150 kW rated transformer)
df["hourly_kwh"] = df["transformer_load_ratio"] * 150.0
baseline_revenue = (df["base_tou_price"] * df["hourly_kwh"]).sum()

print(f"Total Annual Energy Delivered: {df['hourly_kwh'].sum():,.2f} kWh")
print(f"Baseline SCE Annual Revenue:   ${baseline_revenue:,.2f}")


# 3. Dynamic pricing formula: P_dyn = P_SCE * (1 + delta)
def calculate_dynamic_prices(data, delta_surplus, delta_peak):
  multipliers = np.zeros(len(data))

  # State 0: Surplus Discount
  mask_surplus = data["state"] == 0
  multipliers[mask_surplus] = delta_surplus

  # State 2: Peak Stress Proportional Congestion Surcharge
  mask_peak = data["state"] == 2
  load_peak = data.loc[mask_peak, "transformer_load_ratio"]
  mean_peak_load = load_peak.mean() if load_peak.mean() > 0 else 1.0

  # Scale surcharge based on actual real-time transformer stress
  load_scaling = np.clip(load_peak / mean_peak_load, 0.5, 2.0)
  multipliers[mask_peak] = delta_peak * load_scaling

  # Compute tariff and apply regulatory bounds [$0.05, $1.00]
  p_dynamic = data["base_tou_price"].values * (1.0 + multipliers)
  p_dynamic = np.clip(p_dynamic, 0.05, 1.00)

  return p_dynamic, multipliers


# 4. Numerical Optimization for Revenue Neutrality
def objective_function(params):
  delta_surplus, delta_peak = params
  p_dyn, _ = calculate_dynamic_prices(df, delta_surplus, delta_peak)
  dyn_revenue = (p_dyn * df["hourly_kwh"].values).sum()

  # Penalize deviation from baseline revenue + anchor surplus discount near -20%
  revenue_discrepancy = (dyn_revenue - baseline_revenue) ** 2
  discount_anchor = (delta_surplus - (-0.20)) ** 2 * 1e5

  return revenue_discrepancy + discount_anchor


bounds = [(-0.30, -0.05), (0.10, 0.50)]
initial_guess = [-0.15, 0.25]

result = minimize(
    objective_function, initial_guess, bounds=bounds, method="L-BFGS-B"
)
opt_delta_surplus, opt_delta_peak = result.x

print("\n" + "=" * 65)
print("OPTIMAL PRICING MULTIPLIERS FOUND:")
print("=" * 65)
print(f"  - State 0 (Surplus) Discount:      {opt_delta_surplus*100:+.2f}%")
print("  - State 1 (Nominal) Rate:           +0.00% (Baseline SCE)")
print(f"  - State 2 (Peak Stress) Surcharge: {opt_delta_peak*100:+.2f}%")

# 5. Populate optimized columns
df["dynamic_price"], df["price_multiplier"] = calculate_dynamic_prices(
    df, opt_delta_surplus, opt_delta_peak
)
df["dynamic_revenue"] = df["dynamic_price"] * df["hourly_kwh"]
df["baseline_revenue"] = df["base_tou_price"] * df["hourly_kwh"]

optimized_revenue = df["dynamic_revenue"].sum()
rev_diff_pct = (
    (optimized_revenue - baseline_revenue) / baseline_revenue
) * 100

print("\n" + "=" * 65)
print("FINANCIAL VERIFICATION & REVENUE NEUTRALITY:")
print("=" * 65)
print(f"Baseline SCE Revenue:   ${baseline_revenue:,.2f}")
print(f"Dynamic Tariff Revenue: ${optimized_revenue:,.2f}")
print(f"Revenue Discrepancy:    {rev_diff_pct:+.3f}% (Target: within ±1.0%)")

print("\nAverage Price Paid by HMM Regime ($/kWh):")
regime_summary = df.groupby("state_name")[
    ["base_tou_price", "dynamic_price"]
].mean()
for state in ["Surplus", "Nominal", "Peak Stress"]:
  base_p = regime_summary.loc[state, "base_tou_price"]
  dyn_p = regime_summary.loc[state, "dynamic_price"]
  pct = ((dyn_p - base_p) / base_p) * 100
  print(
      f"  - {state:<12}: Baseline = ${base_p:.3f}  -->  Dynamic = ${dyn_p:.3f}"
      f" ({pct:+.1f}%)"
  )

# 6. Save deliverable
output_csv = "hourly_pricing_optimized.csv"
df.to_csv(output_csv, index=False)

print("\n" + "=" * 65)
print(f"SUCCESS: Certified Phase 4.4 output saved to '{output_csv}'")
print("=" * 65)