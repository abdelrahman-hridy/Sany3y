import numpy as np
import pandas as pd

# ============================================================
# 1. General Configuration
# ============================================================
TOTAL_RECORDS = 10000
RANDOM_SEED = 42
OUTPUT_FILE = "sany3y_synthetic_dataset_v1.csv"

rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
# 2. Service catalog: BASE LABOR PRICES ONLY (EGP)
#    Excludes materials, spare parts, and equipment costs.
# ============================================================
services_config = {
    "Plumbing": {
        "Fix_Faucet": 150,
        "Unclog_Sink": 140,
        "Install_Heater": 250,
        "Install_Washing_Machine": 180,
        "Filter_Maintenance": 120,
        "Fix_Minor_Leak": 130,
    },
    "Electricity": {
        "Fix_Short_Circuit": 200,
        "Change_Switch_Socket": 90,
        "Install_Chandelier": 180,
        "Install_Ceiling_Fan": 160,
        "Install_Exhaust_Fan": 130,
        "Install_Led_Strip": 140,
    },
    "HVAC": {
        "AC_Clean_And_Wash": 300,
        "AC_Freon_Refill": 650,
        "AC_Capacitor_Repair": 250,
        "AC_Water_Leak_Fix": 220,
        "AC_Single_Install": 450,
    },
    "Carpentry": {
        "Fix_Door_Lock": 140,
        "Fix_Door_Hinges": 110,
        "Trim_Wooden_Door": 160,
        "Assemble_IKEA_Furniture": 250,
        "Install_Curtains": 130,
    },
    "Painting": {
        "Wall_Moisture_Repair": 250,
        "Paint_Single_Room": 500,
        "Install_Decor_Wall": 350,
    },
}

# ============================================================
# 3. Zone configuration (Initial Market Hypothesis)
#    To be calibrated later with real market quotations.
# ============================================================
zone_classes = ["Class_A", "Class_B", "Class_C"]
zone_probabilities = [0.25, 0.45, 0.30]

zone_multipliers = {
    "Class_A": 1.30,
    "Class_B": 1.10,
    "Class_C": 1.00,
}

# ============================================================
# 4. Other pricing assumptions
# ============================================================
PEAK_HOUR_PROBABILITY = 0.35
WEEKEND_PROBABILITY = 0.25

PEAK_HOUR_INCREASE = 0.15
WEEKEND_INCREASE = 0.10

MIN_DISTANCE_KM = 1.0
MAX_DISTANCE_KM = 20.0
DISTANCE_FREE_THRESHOLD_KM = 1.5
DISTANCE_LABOR_ADJUSTMENT_PER_KM = 5.0

NOISE_STD_PERCENT = 0.08
PRICE_FLOOR_PERCENT_OF_BASE = 0.90

# ============================================================
# 5. Validate configuration
# ============================================================
if TOTAL_RECORDS <= 0:
    raise ValueError("TOTAL_RECORDS must be positive.")

if not np.isclose(sum(zone_probabilities), 1.0):
    raise ValueError("Zone probabilities must sum to 1.")

if set(zone_classes) != set(zone_multipliers):
    raise ValueError("Every zone class must have exactly one multiplier.")

for category, subcategories in services_config.items():
    if not subcategories:
        raise ValueError(f"{category} has no subcategories.")
    if any(price <= 0 for price in subcategories.values()):
        raise ValueError(f"{category} contains a non-positive base labor price.")

# ============================================================
# 6. Create balanced main-category sampling (2,000 per category)
# ============================================================
categories = list(services_config.keys())
base_count = TOTAL_RECORDS // len(categories)
remainder = TOTAL_RECORDS % len(categories)

category_schedule = []
for index, category in enumerate(categories):
    count = base_count + (1 if index < remainder else 0)
    category_schedule.extend([category] * count)

rng.shuffle(category_schedule)

# ============================================================
# 7. Generate synthetic records
# ============================================================
data_rows = []

for category in category_schedule:
    subcategory_names = list(services_config[category].keys())
    sub_category = rng.choice(subcategory_names)
    base_labor_price = services_config[category][sub_category]

    zone = rng.choice(zone_classes, p=zone_probabilities)
    zone_multiplier = zone_multipliers[zone]

    distance_km = round(float(rng.uniform(MIN_DISTANCE_KM, MAX_DISTANCE_KM)), 1)
    distance_labor_adjustment = (
        max(0.0, distance_km - DISTANCE_FREE_THRESHOLD_KM) * DISTANCE_LABOR_ADJUSTMENT_PER_KM
    )

    is_peak_hour = int(rng.random() < PEAK_HOUR_PROBABILITY)
    is_weekend = int(rng.random() < WEEKEND_PROBABILITY)

    # Calculate final price
    price = (base_labor_price * zone_multiplier) + distance_labor_adjustment
    if is_peak_hour:
        price *= (1 + PEAK_HOUR_INCREASE)
    if is_weekend:
        price *= (1 + WEEKEND_INCREASE)

    # Add realistic market variance (noise)
    noise = rng.normal(0, price * NOISE_STD_PERCENT)
    final_price = max(base_labor_price * PRICE_FLOOR_PERCENT_OF_BASE, price + noise)

    data_rows.append({
        "Service_Category": category,
        "Sub_Category": sub_category,
        "Zone_Class": zone,
        "Distance_KM": distance_km,
        "Is_Peak_Hour": is_peak_hour,
        "Is_Weekend": is_weekend,
        "Final_Price_EGP": round(final_price, 2)
    })

df = pd.DataFrame(data_rows)

# ============================================================
# 8. Sanity Checks & Export
# ============================================================
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Dataset V1 generated successfully: '{OUTPUT_FILE}'")
print(f"📊 Total Records: {len(df)}")
print(f"❓ Missing Values: {df.isnull().sum().sum()}")
print(f"👯 Exact Duplicates: {df.duplicated().sum()}")
print("\n--- Category Distribution ---")
print(df["Service_Category"].value_counts())
print("\n--- Price Summary (EGP) ---")
print(df["Final_Price_EGP"].describe().round(2))