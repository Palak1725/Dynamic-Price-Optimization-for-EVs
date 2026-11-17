Architecture — Dynamic Price Optimization of EV Charging
1. Project Overview
Project Name: Dynamic Price Optimization of EV Charging
Domain: Artificial Intelligence, Machine Learning and Data Science
Primary Dataset: Caltech ACN-Data (Adaptive Charging Network)
Programming Language: Python
Core Objective:
To develop a data-driven EV charging price optimization system that analyzes historical EV charging sessions, identifies charging demand patterns, detects different operational conditions using a Hidden Markov Model (HMM), and calculates an optimized charging price based on demand, electricity procurement costs, and grid constraints.
The project consists of five major phases:
	Data Acquisition and Raw Ingestion
	Data Cleaning and Preprocessing
	Feature Engineering and Time-Series Construction
	Hidden Markov Model Development
	Dynamic Pricing and Optimization
 
2. High-Level System Architecture
The system follows a sequential data processing and optimization architecture.
             CALTECH ACN DATASET
                      |
                      v
           PHASE 1: DATA ACQUISITION
                      |
             Raw Data Ingestion
             Schema Validation
             Data Profiling
                      |
                      v
          PHASE 2: DATA PREPROCESSING
                      |
             Timestamp Standardization
             Missing Value Handling
             Duplicate Removal
             Outlier Detection
                      |
                      v
        PHASE 3: FEATURE ENGINEERING
                      |
             Hourly Aggregation
             Occupancy Calculation
             Load Features
             Time-Based Features
             Feature Normalization
                      |
                      v
          PHASE 4: HIDDEN MARKOV MODEL
                      |
             Gaussian HMM
             Baum-Welch Training
             Viterbi Decoding
             Operational State Detection
                      |
                      v
        PHASE 5: PRICE OPTIMIZATION
                      |
             Demand Prediction
             Price Elasticity
             Grid Penalty Calculation
             Constrained Optimization
                      |
                      v
                 FINAL OUTPUT
                      |
             Optimal Charging Price
                   (Rs/kWh)
 
3. Dataset Architecture
3.1 Primary Dataset: Caltech ACN-Data
The project uses the Caltech Adaptive Charging Network (ACN) dataset containing historical EV charging session information.
Important dataset attributes include:
Attribute	Description
connectionTime	Time when the vehicle connects
disconnectionTime	Time when the vehicle disconnects
doneChargingTime	Time when charging is completed
kWhDelivered	Total energy delivered
sessionID	Unique charging session identifier
stationID	Charging station identifier
userID	User identifier, when available
userInputs	Additional session information, when available


3.2 Dataset Processing
The raw ACN dataset is converted into structured tabular data using Python and Pandas.
The processing pipeline includes:
	JSON extraction and parsing.
	Session-level data extraction.
	Timestamp conversion.
	Data quality validation.
	Missing-value analysis.
	Hourly aggregation.
3.3 External Configuration Data
The system also uses external configuration for electricity procurement costs.
Configuration file: config/grid_tariffs.json
The proposed Time-of-Day tariff structure is:
Time Period	Electricity Cost
Off-Peak (22:00–06:00)	₹5.50/kWh
Normal (06:00–17:00)	₹7.00/kWh
Evening Peak (17:00–22:00)	₹10.50/kWh
These values are configuration inputs and should be verified against the applicable official tariff schedule before real-world deployment.
 
4. Phase-Wise System Architecture
PHASE 1 — Data Acquisition and Raw Ingestion
Objective: Collect and store raw EV charging session data for further processing.
Input:
	Caltech ACN JSON dataset.
Main Operations:
	Dataset identification and extraction.
	JSON data ingestion.
	Session record extraction.
	Schema validation.
	Data type verification.
	Raw data storage.
Important Attributes:
	sessionID
	stationID
	connectionTime
	disconnectionTime
	doneChargingTime
	kWhDelivered
Output:
Raw structured EV charging session data.
Expected Storage:
data/
└── raw/
    ├── acndata_sessions.json
    └── acndata_sessions (1).json
PHASE 2 — Data Cleaning and Preprocessing
Objective: Improve data quality and prepare reliable session records for feature engineering.
Input: Raw Caltech ACN session data.
Main Operations:
	Timestamp standardization.
	Timezone normalization.
	Missing-value detection and handling.
	Duplicate record identification.
	Invalid timestamp correction.
	Zero-energy session filtering.
	Statistical outlier detection using IQR and Z-score.
	Session duration calculation.
Derived Variables:
Session Duration:
"Session Duration (minutes)"=("Disconnect Time" -"Connection Time" )/60


Active Charging Duration:
"Active Charging Minutes"=("Done Charging Time" -"Connection Time" )/60

Idle Duration:
"Idle Duration Minutes"=("Disconnect Time" -"Done Charging Time" )/60

Charging Rate:
"Charging Rate (kW)"="Energy Delivered (kWh)" /"Active Charging Time / 60" 

Cleaned and validated EV charging session dataset.
PHASE 3 — Feature Engineering and Time-Series Construction
Objective: Transform individual EV charging sessions into a continuous hourly time-series dataset.
Input: Cleaned session dataset from Phase 2.
Main Operations:
	Hourly resampling.
	Aggregate charging load calculation.
	Occupancy rate calculation.
	Arrival rate calculation.
	Transformer load ratio calculation.
	Rolling statistical features.
	Calendar feature extraction.
	Cyclical time encoding.
	Feature scaling.
	Chronological train-validation-test splitting.
Important Features:
Feature	Purpose
Aggregate Power	Total charging power
Occupancy Rate	Charging port utilization
Arrival Rate	Number of new charging arrivals
Transformer Load Ratio	Transformer utilization
Hour of Day	Hourly demand pattern
Day of Week	Weekly demand pattern
Is Weekend	Weekend indicator
Rolling Mean	Demand trend
Rolling Standard Deviation	Demand variability
Occupancy Rate:
Occupancy Rate =
Occupied Charging Ports / Total Charging Ports
Transformer Load Ratio:
TLR =
Aggregate Active Power / Transformer Capacity
Cyclical Time Encoding:
Hour_sin = sin(2 * pi * hour / 24)

Hour_cos = cos(2 * pi * hour / 24)
Feature Matrix:
The observation matrix is represented as:
O = [Aggregate Power,
     Occupancy Rate,
     Arrival Rate,
     Hour_sin,
     Hour_cos]
Output:
Hourly feature dataset ready for HMM development.
PHASE 4 — Hidden Markov Model Development
Objective: Identify hidden operational charging demand regimes using a Gaussian Hidden Markov Model.
Input: Normalized hourly feature matrix from Phase 3.
Technology:
	hmmlearn
	GaussianHMM
	NumPy
	Scikit-learn
Model Configuration:
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=200,
    random_state=42
)
Three Hidden Operational States:
Hidden State	Regime	Interpretation
S1	Surplus	Low charging load
S2	Baseline	Normal charging demand
S3	Peak	High charging load and grid stress
Main Operations:
	Observation matrix construction.
	Gaussian HMM initialization.
	Baum-Welch model training.
	Log-likelihood convergence monitoring.
	Model selection using AIC and BIC.
	Viterbi state decoding.
	State transition probability analysis.
	Operational regime interpretation.
HMM Components:
	Initial State Probability (π)
	Transition Probability Matrix (A)
	Emission Probability Distribution (B)
Output:
	Trained Gaussian HMM model.
	Predicted operational states.
	State transition matrix.
	State probabilities and confidence information.
PHASE 5 — Dynamic Pricing and Optimization
Objective: Calculate the optimal EV charging price based on operational conditions, demand elasticity, electricity procurement cost, and grid safety constraints.
Input:
	Predicted HMM operational state.
	Hourly demand features.
	Electricity procurement cost.
	Base charging price.
	Grid capacity constraints.
Main Operations:
	Regime-conditioned price elasticity.
	Demand prediction.
	Net margin calculation.
	Grid overload penalty calculation.
	Constrained price optimization.
	Regulatory and ethical pricing safeguards.
Regime-Based Price Elasticity:
State	Operational Regime	Assumed Elasticity
S1	Surplus	-1.3
S2	Baseline	-0.8
S3	Peak	-0.3
These elasticity coefficients are initial model assumptions unless estimated and validated using observed price-response data.
Demand Function:
Q(P | S) =
Q_base * (P / P_base) ^ epsilon
Where:
	Q = Predicted charging demand.
	P = Dynamic charging price.
	Q_base = Baseline demand.
	P_base = Baseline charging price.
	epsilon = Price elasticity for the operational state.
Optimization Objective:
Maximize:

Profit =
(P - C_wholesale) * Q(P | S)
- Grid Overload Penalty
Grid Overload Penalty:
If Load <= Safe Capacity:

    Penalty = 0

If Load > Safe Capacity:

    Penalty = kappa * (Load - Safe Capacity)^2
Optimization Technology:
	SciPy
	scipy.optimize.minimize
	SLSQP
	NumPy
Pricing Constraints:
P_min = C_wholesale + delta

P_min <= P_optimal <= P_max
A lifeline pricing safeguard is also included in the proposed design for vehicles with SOC below 15%, limiting the dynamic tariff to the base rate.
Final Output:
	Optimal charging price (₹/kWh).
	Predicted demand.
	Expected net margin.
	Grid overload penalty.
	Recommended pricing regime.
 
5. Technology Stack
Technology	Application
Python	Main programming language
Pandas	Data manipulation
NumPy	Numerical computation
SciPy	Statistical analysis and optimization
Scikit-learn	Feature scaling and preprocessing
hmmlearn	Gaussian Hidden Markov Model
Joblib	Model serialization
Matplotlib	Data visualization
Seaborn	Statistical visualization
JSON	Dataset and tariff configuration
 
6. Proposed Project Directory Structure
The following is a logical project structure. Keep your actual filenames and directories if they are already different.
EV_charge/
│
├── data/
│   ├── raw/
│   │   └── Caltech_ACN/
│   │
│   └── processed/
│
├── config/
│   └── grid_tariffs.json
│
├── src/
│   ├── phase1_data_acquisition/
│   │
│   ├── phase2_data_cleaning/
│   │
│   ├── phase3_feature_engineering/
│   │
│   ├── phase4_hmm_model/
│   │
│   └── phase5_price_optimization/
│
├── models/
│
├── outputs/
│   ├── visualizations/
│   ├── predictions/
│   └── optimization_results/
│
├── requirements.txt
│
├── README.md
│
└── architecture.md
 
7. End-to-End Data Flow
The complete system follows this sequence:
	Collect historical EV charging session records from Caltech ACN.
	Extract session information from the raw JSON files.
	Clean and preprocess the session records.
	Generate hourly time-series features.
	Normalize the feature matrix.
	Train the Gaussian Hidden Markov Model.
	Identify the current operational charging regime.
	Estimate demand based on the identified regime.
	Apply the corresponding price elasticity.
	Calculate the optimal charging price using constrained optimization.
	Generate the final pricing recommendation.
 
8. Important Design Considerations
	Raw datasets should remain unchanged.
	Data cleaning should be reproducible.
	Missing and invalid values should be handled consistently.
	Train-test splitting should preserve chronological order.
	Feature scaling should avoid data leakage.
	HMM states should be interpreted using their observed feature characteristics.
	Electricity procurement tariffs should remain separate from the HMM observation vector.
	Price elasticity assumptions should be validated before practical use.
	Pricing should respect grid capacity and ethical pricing safeguards.
Dataset limitation: Caltech ACN session records provide charging-session information. They do not automatically provide every external variable required by the pricing engine, such as local DISCOM tariffs, sanctioned transformer capacity, or observed consumer response to different prices. These must be separately configured, measured, or explicitly simulated.
 
9. Expected Final System Outcome
The final system aims to connect historical EV charging behavior with intelligent pricing decisions.
Input: Historical EV charging session data.
Processing: Data cleaning, hourly feature engineering, operational state detection, and demand estimation.
Optimization: Dynamic price calculation using economic and grid safety constraints.
Output: An optimized EV charging tariff expressed in ₹/kWh.
The proposed architecture provides a modular five-phase framework that can be extended with real-time charging data, additional grid information, and a pricing dashboard in future development.
<img width="468" height="638" alt="image" src="https://github.com/user-attachments/assets/0fabacb8-675d-45b8-a17f-1d118d879767" />
