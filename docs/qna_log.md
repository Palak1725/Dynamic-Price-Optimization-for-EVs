# Project Design Decisions & Technical Q&A

This document records architectural decisions, engineering edge cases, and clarifications resolved during development.

---

## Phase 1: Repository Initialisation and Appropriate Data Selection
### Q1: 
* **Raised by:** 
* **Resolution:** 

---

## Phase 2: Data Cleaning & Preprocessing
Raised by:Ankush

Q1: What is the Caltech ACN dataset, and what does it represent?

Answer: It contains real-world EV charging logs from 
Caltech's parking garage in Pasadena, California, across the full year of 2019. It captures workplace commuter charging across 54 physical Level 2 charging stalls, where vehicles stay parked for hours during work shifts.   

Q2: Why did we have two separate JSON files (acndata_sessions.json and acndata_sessions (1).json)?

Answer: The full 2019 data exceeded the Caltech API export size limit. It was split into two exports: File 1 (7,440 sessions) and File 2 (3,177 sessions), totaling 10,617 raw session logs that were normalized and concatenated.

Q3: How did we verify that merging the two files didn't create duplicate sessions?

Answer: We executed df['sessionID'].duplicated().sum(), which returned 0. Every record has a globally unique primary key.

Q4: Why is chronological sorting (connectionTime) strictly necessary?

Answer: Time-series feature engineering and 1-hour interval resampling require strictly forward-flowing time. Chronological sorting prevents temporal leakage (using future information to predict the past) and guarantees causal consistency.Section 2: Timezones & Temporal Integrity

Q5: Why do we convert timestamps to America/Los_Angeles?

Answer: Caltech is physically located in Pasadena, California. Converting to local Pacific Time is mandatory to align charging events with actual daytime work hours, identify morning commuter arrival peaks, and synchronize sessions with local Southern California Edison (SCE) Time-of-Use tariffs. 

Q6: Why can't we use Indian Standard Time (IST)?

Answer: IST is 12.5 to 13.5 hours ahead of California. Using IST inverts day and night—turning an 8:30 AM California morning office arrival into a 10:00 PM late-night event, completely corrupting diurnal behavior patterns.

Q7: What is the Dual-Timezone architecture, and why do we use both UTC and Pacific Time?

Answer:UTC (Mathematical Spine): Unaffected by clock shifts; guarantees an unbroken, gap-free continuous timeline of exactly 8,760 hours for 2019.Pacific Time (Behavioral Projection): Used to extract local_hour (0–23), day_of_week, and TOU tariffs to match local human schedules.

Q8: What is the difference between PST, PDT, and DST?

Answer:DST (Daylight Saving Time): The practice of advancing clocks by 1 hour during warmer months.PST (Pacific Standard Time): California's winter time zone ($\text{UTC}-8$).PDT (Pacific Daylight Time): California's summer time zone while DST is active ($\text{UTC}-7$).Pandas handles this automatically via America/Los_Angeles.Section 3: Duration Disaggregation & Electrical Formulas

Q9: What is Total Session Duration, and how is it calculated?

Answer: The total duration a vehicle occupies a charging stall:
   $$\text{Session Duration (mins)} = \frac{\text{disconnectTime} - \text{connectionTime}}{60}$$
It measures physical parking congestion and stall availability.   

Q10: What is Idle Duration, and why is it called "stall blocking"?

Answer: The time an EV remains plugged in after its battery is full (zero power draw):
   $$\text{Idle Duration (mins)} = \frac{\text{disconnectTime} - \text{doneChargingTime}}{60}$$
It identifies "stall blocking," where fully charged cars prevent other drivers from using the charger.  

Q11: What is Active Charging Duration, and why is it calculated separately?

Answer: The exact duration power is actively flowing into the vehicle:
   $$\text{Active Duration (mins)} = \frac{\text{doneChargingTime} - \text{connectionTime}}{60}$$
Separating active time from idle time prevents phantom load errors—ensuring the grid model does not assume a vehicle is drawing power during its entire 8-hour parking stay.   

Q12: How is the Implied Charging Rate (charging_rate_kw) calculated?

Answer: The average power drawn during active charging:
   $$\text{Charging Rate (kW)} = \frac{\text{kWhDelivered}}{\text{active\_charging\_minutes} / 60}$$

Q13: Why did we clip active_charging_minutes to a minimum of 3.0 minutes?

Answer: To eliminate division-by-zero and extreme mathematical spikes. A 10-second connection would otherwise yield an impossible power rate of over $300\text{ kW}$ on a standard $7\text{ kW}$ Level 2 charger.Section 4: Data Anomalies, Cleaning Decisions & Missing Values

Q14: How did we handle missing or corrupted doneChargingTime timestamps?

Answer: When doneChargingTime was null, earlier than connectionTime, or later than disconnectTime, it was imputed to equal disconnectTime. This conservatively assumes continuous trickle power draw instead of discarding valid charging sessions.

Q15: Why are there 1,532 missing values in userID and userInputs, and why didn't we drop them?

Answer: These represent anonymous guest drivers who plugged in without registering on the mobile app. Their energy meters and timestamps are 100% complete. Because we optimize station-level load rather than individual user profiles, dropping them would omit 14.5% of real grid electrical demand.

Q16: Why did we filter out sessions delivering less than 0.5 kWh?

Answer: Sessions under $0.5\text{ kWh}$ are aborted communication handshakes, declined payment cards, or instant stall disconnects. They represent telemetry noise rather than true charging events.

Q17: Why did we drop sessions longer than 24 hours (1,440 minutes)?

Answer: The dataset contained abandoned vehicles parked for up to 9 days (214 hours) over university holidays. These are long-term vehicle storage events that corrupt daily commuter occupancy profiles and Markov state transitions.

Q18: Why did we cap implied charging rates at $\le 25\text{ kW}$?

Answer: Caltech exclusively uses J1772 Level 2 AC chargers, which have a physical hardware ceiling of $\sim 19.2\text{--}22\text{ kW}$. Any reading above $25\text{ kW}$ is sensor current drift or data corruption.

Q19: What is the exact accounting of rows dropped during Phase 2?

Answer: We ingested 10,617 raw rows, removed 78 invalid rows (77 multi-day outliers $> 24$ hours, 1 sensor power spike $> 25\text{ kW}$, and handshake drops), leaving exactly 10,539 clean rows (99.26% retention).Section 5: Statistical Methods: Outliers, IQR & Z-Scores

Q20: What is the Interquartile Range (IQR) method?

Answer: A non-parametric method based on percentiles measuring the middle 50% spread: $\text{IQR} = Q_3 - Q_1$. Data points outside $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$ are flagged as outliers.  

Q21: What is a Z-Score?

Answer: A parametric metric indicating how many standard deviations a value lies from the mean:
   $$z = \frac{x - \mu}{\sigma}$$
Values with $\vert{}z\vert{} > 3.0$ denote extreme statistical deviations.   

Q22: Why does Z-Score fail when applied to raw, uncleaned EV charging data?

Answer: Z-scores assume a symmetric normal distribution. Raw EV data is heavily right-skewed with long tails (e.g., 200-hour sessions). These extremes inflate both the mean and standard deviation, shrinking calculated Z-scores and masking genuine anomalies.   

Q23: Why did we use IQR for outlier detection but Z-Score for feature standardization?

Answer: IQR uses medians and quartiles, making it robust against skewed data for initial outlier removal. Once extreme noise was removed, Z-score transformation was applied to standardize features onto a zero-mean, unit-variance scale for downstream machine learning.   Section 6: Final Verification & Pipeline Handoff

Q24: What are the final dimensions and summary statistics of the cleaned dataset?

Answer: 10,539 rows $\times$ 27 columns.Session Duration: Mean $345.6\text{ min}$ (5.76 h), Median $345.7\text{ min}$, Max $1,435.2\text{ min}$ (23.9 h).Active Charging: Mean $173.1\text{ min}$ (2.88 h), Median $120.3\text{ min}$.Idle Duration: Mean $172.5\text{ min}$ (2.87 h), Median $93.2\text{ min}$.Energy Delivered: Mean $8.86\text{ kWh}$, Max $75.53\text{ kWh}$.Charging Rate: Mean $3.40\text{ kW}$, Max $20.13\text{ kW}$.

Q25: What sanity checks prove that Phase 2 was completed correctly?

Answer:Zero duplicate session IDs.Zero null values in operational columns.Strict chronological monotonicity (connectionTime.is_monotonic_increasing == True).100% sequence validity: $\text{connectionTime} \le \text{doneChargingTime} \le \text{disconnectTime}$.Exact duration additivity: $\text{Session Duration} = \text{Active Duration} + \text{Idle Duration}$.

Q26: How does Phase 2 directly connect to Phase 3 and Phase 4?

Answer: Phase 2 cleans and separates active power from idle parking. This allows Phase 3 to resample sessions into an 8,760-hour continuous timeline to compute Occupancy Rate ($OR_t$) and Transformer Load Ratio ($TLR_t$), which train the Gaussian HMM to classify grid stress regimes for Phase 4 dynamic pricing.
---

## Phase 3: Time-Series Resampling & HMM Modeling
# EV Charging Optimization — Project Q&A

### Q1. How can we convert charging session data into hourly data?
**Raised by:** Anshika Kumari
**Solution:** I created an hourly timeline to convert the charging session data into a format suitable for hourly analysis. This helps us understand charging activity hour by hour. For a complete non-leap year, the timeline contains 8,760 hours.

### Q2. How can we convert UTC timestamps into local time for charging demand analysis?
**Raised by:** Anshika Kumari
**Solution:** I converted the UTC timestamps into the local timezone (`America/Los_Angeles`). This helps identify the correct local hour, day of the week, and whether the day is a weekday or weekend, which are important for analysing charging demand.

### Q3. How can we calculate the transformer load ratio for each hour?
**Raised by:** Anshika Kumari
**Solution:** I estimated the charging load for each hour by calculating how long vehicles were actively charging during that hour and using their charging rates to estimate energy consumption. I then divided the estimated hourly load by the assumed transformer capacity of 150 kW to calculate the transformer load ratio.

### Q4. How can we assign time-of-use (TOU) prices to each hour?
**Raised by:** Anshika Kumari
**Solution:** I assigned electricity prices to each hourly record based on the local time and whether the day was a weekday or weekend. This helps represent how electricity prices may vary throughout the day. The price values used in the project are predefined assumptions and should be verified against the actual electricity tariff before being treated as official prices.
