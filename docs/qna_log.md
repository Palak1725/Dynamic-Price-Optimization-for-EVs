# Project Design Decisions & Technical Q&A

This document records architectural decisions, engineering edge cases, and clarifications resolved during development.

---

## Phase 1: Repository Initialisation and Appropriate Data Selection
### Q1: 
* **Raised by:** 
* **Resolution:** 

---

## Phase 2: Data Cleaning & Preprocessing

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
