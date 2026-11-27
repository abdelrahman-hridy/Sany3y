# ==========================================
# Interactive Price Prediction CLI
# ==========================================

import joblib
import numpy as np
import pandas as pd

model = int(input("model (1 = Random Forest, 0 = XGBoost): ").strip())
MODEL_PATH =  "models/sany3y_hybrid_model.joblib" if model == 1 else "models/hybrid_xgboost_model.joblib"
artifact = joblib.load(MODEL_PATH)

base_pipeline = artifact["base_pipeline"]
context_features = artifact["context_features"]
FREE_KM = artifact["free_km"]
RATE_PER_KM = artifact["rate_per_km"]

SERVICES = {
    "Carpentry": [
        "Assemble_IKEA_Furniture",
        "Fix_Door_Hinges",
        "Fix_Door_Lock",
        "Install_Curtains",
        "Trim_Wooden_Door",
    ],
    "Electricity": [
        "Change_Switch_Socket",
        "Fix_Short_Circuit",
        "Install_Ceiling_Fan",
        "Install_Chandelier",
        "Install_Exhaust_Fan",
        "Install_Led_Strip",
    ],
    "HVAC": [
        "AC_Capacitor_Repair",
        "AC_Clean_And_Wash",
        "AC_Freon_Refill",
        "AC_Single_Install",
        "AC_Water_Leak_Fix",
    ],
    "Painting": [
        "Install_Decor_Wall",
        "Paint_Single_Room",
        "Wall_Moisture_Repair",
    ],
    "Plumbing": [
        "Filter_Maintenance",
        "Fix_Faucet",
        "Fix_Minor_Leak",
        "Install_Heater",
        "Install_Washing_Machine",
        "Unclog_Sink",
    ],
}

ZONES = ["Class_A", "Class_B", "Class_C"]


def select_option(title, options):
    print(f"\n{title}:")
    for idx, item in enumerate(options, start=1):
        print(f"  {idx}. {item}")
    while True:
        choice = input("Select number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


while True:
    print("\n" + "=" * 40)
    print("Sany3y Smart Pricing Simulator")
    print("=" * 40)

    category = select_option("Service Category", list(SERVICES.keys()))
    sub_category = select_option("Sub Category", SERVICES[category])
    zone = select_option("Zone Class", ZONES)

    distance = float(input("\nDistance (KM): ").strip())
    is_peak = int(input("Is Peak Hour? (1 = Yes, 0 = No): ").strip())
    is_weekend = int(input("Is Weekend? (1 = Yes, 0 = No): ").strip())
    

    request_df = pd.DataFrame([{
        "Service_Category": category,
        "Sub_Category": sub_category,
        "Zone_Class": zone,
        "Distance_KM": distance,
        "Is_Peak_Hour": is_peak,
        "Is_Weekend": is_weekend,
        
    }])

    base_price = float(base_pipeline.predict(request_df[context_features])[0])
    distance_fee = float(np.maximum(0.0, distance - FREE_KM) * RATE_PER_KM)
    final_price = base_price + distance_fee

    print("\n" + "-" * 40)
    print(f"Base Service Price : {base_price:.2f} EGP")
    print(f"Distance Fee       : {distance_fee:.2f} EGP")
    print(f"Predicted Price    : {final_price:.2f} EGP")
    print("-" * 40)

    if input("\nTest another request? (y/n): ").strip().lower() != "y":
        break