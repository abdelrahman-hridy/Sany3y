# ==========================================
# 1. Import Libraries
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ==========================================
# 2. Load Dataset
# ==========================================

DATA_PATH = "sany3y_synthetic_dataset_v1.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ==========================================
# 3. Data Quality Validation
# ==========================================

TARGET = "Final_Price_EGP"

categorical_features = [
    "Service_Category",
    "Sub_Category",
    "Zone_Class"
]

numerical_features = [
    "Is_Peak_Hour",
    "Is_Weekend"
]

context_features = categorical_features + numerical_features

expected_columns = context_features + ["Distance_KM"]

missing_columns = [
    col for col in expected_columns + [TARGET]
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("\n========== DATA QUALITY ==========")

print("\nMissing Values:")
print(df[expected_columns + [TARGET]].isnull().sum())

if df[expected_columns + [TARGET]].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

print("\nDuplicate Rows:", df.duplicated().sum())

if (df["Distance_KM"] < 0).any():
    raise ValueError("Distance cannot be negative.")

if (df[TARGET] <= 0).any():
    raise ValueError("Target price must be positive.")

for col in ["Is_Peak_Hour", "Is_Weekend"]:
    if not df[col].isin([0, 1]).all():
        raise ValueError(f"{col} must contain only 0 and 1.")

print("\nData Quality Validation Passed!")


# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df[expected_columns].copy()

y = df[TARGET].copy()


# ==========================================
# 5. Train / Validation / Test Split
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\n========== DATA SPLIT ==========")

print(f"Training:   {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")


# ==========================================
# 6. Hybrid Distance Pricing Rule
# ==========================================

FREE_KM = 1.5
RATE_PER_KM = 5.0


def calc_distance_fee(distances):

    return (
        np.maximum(
            0.0,
            np.asarray(distances, dtype=float) - FREE_KM
        )
        * RATE_PER_KM
    )


def predict_hybrid(model, X_data):

    base_price = model.predict(
        X_data[context_features]
    )

    distance_fee = calc_distance_fee(
        X_data["Distance_KM"]
    )

    return base_price + distance_fee


# Calculate Base Service Price

y_train_base = (
    y_train - calc_distance_fee(
        X_train["Distance_KM"]
    )
)


# ==========================================
# 7. Preprocessing Pipeline
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ],
    remainder="drop"
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="mae",
                random_state=42,
                n_jobs=1,
                tree_method="hist"
            )
        )
    ]
)


# ==========================================
# 8. Hyperparameter Search Space
# ==========================================

param_distributions = {

    "model__n_estimators": [
        200, 300, 400, 500, 600, 700
    ],

    "model__max_depth": [
        2, 3, 4, 5, 6
    ],

    "model__learning_rate": [
        0.01, 0.03, 0.05, 0.1
    ],

    "model__min_child_weight": [
        1, 3, 5, 7
    ],

    "model__subsample": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],

    "model__colsample_bytree": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],

    "model__gamma": [
        0, 0.1, 0.3, 0.5
    ],

    "model__reg_alpha": [
        0, 0.01, 0.1, 1
    ],

    "model__reg_lambda": [
        1, 2, 5, 10
    ]
}


# ==========================================
# 9. Hybrid Cross Validation Scoring
# ==========================================

def hybrid_cv_scorer(estimator, X_eval, y_base_eval):

    base_pred = estimator.predict(
        X_eval[context_features]
    )

    distance_fee = calc_distance_fee(
        X_eval["Distance_KM"]
    )

    actual_final = (
        np.asarray(y_base_eval) + distance_fee
    )

    predicted_final = base_pred + distance_fee

    return -mean_absolute_percentage_error(
        actual_final,
        predicted_final
    )


cv_strategy = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==========================================
# 10. Randomized Search
# ==========================================

random_search = RandomizedSearchCV(
    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=40,

    scoring=hybrid_cv_scorer,

    cv=cv_strategy,

    random_state=42,

    n_jobs=-1,

    verbose=1,

    return_train_score=True
)

print("\n========== STARTING XGBOOST SEARCH ==========")

random_search.fit(
    X_train,
    y_train_base
)

best_model = random_search.best_estimator_


print("\n========== BEST PARAMETERS ==========")

print(random_search.best_params_)

print(
    f"\nBest CV Final Hybrid MAPE: "
    f"{-random_search.best_score_ * 100:.2f}%"
)


# ==========================================
# 11. Model Evaluation
# ==========================================

def evaluate_model(model, X_data, y_true, dataset_name):

    y_pred = predict_hybrid(
        model,
        X_data
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mape = (
        mean_absolute_percentage_error(
            y_true,
            y_pred
        ) * 100
    )

    wape = (
        np.sum(
            np.abs(
                np.asarray(y_true) - y_pred
            )
        )
        / np.sum(np.asarray(y_true))
    ) * 100

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(f"\n{dataset_name} Results")
    print("-" * 40)

    print(f"MAE:  {mae:.2f} EGP")
    print(f"RMSE: {rmse:.2f} EGP")
    print(f"MAPE: {mape:.2f}%")
    print(f"WAPE: {wape:.2f}%")
    print(f"R2:   {r2:.4f}")

    return y_pred, {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape,
        "R2": r2
    }


print("\n========== MODEL EVALUATION ==========")

y_train_pred, train_metrics = evaluate_model(
    best_model,
    X_train,
    y_train,
    "Training"
)

y_val_pred, val_metrics = evaluate_model(
    best_model,
    X_val,
    y_val,
    "Validation"
)

y_test_pred, test_metrics = evaluate_model(
    best_model,
    X_test,
    y_test,
    "Final Test"
)


# ==========================================
# 12. Segment Performance Analysis
# ==========================================

test_results = X_test.copy()

test_results["Actual_Price"] = y_test.values

test_results["Predicted_Price"] = y_test_pred

test_results["Absolute_Error"] = np.abs(
    test_results["Actual_Price"]
    - test_results["Predicted_Price"]
)

test_results["Absolute_Percentage_Error"] = (
    test_results["Absolute_Error"]
    / test_results["Actual_Price"]
) * 100


def segment_analysis(data, group_column):

    return data.groupby(group_column).agg(

        Samples=("Actual_Price", "count"),

        Average_Actual=("Actual_Price", "mean"),

        Average_Predicted=("Predicted_Price", "mean"),

        MAE=("Absolute_Error", "mean"),

        MAPE=("Absolute_Percentage_Error", "mean")

    ).round(2)


print("\n========== PERFORMANCE BY CATEGORY ==========")

category_results = segment_analysis(
    test_results,
    "Service_Category"
)

print(category_results)


print("\n========== PERFORMANCE BY SUB-CATEGORY ==========")

subcategory_results = segment_analysis(
    test_results,
    "Sub_Category"
)

print(subcategory_results)


print("\n========== PERFORMANCE BY ZONE ==========")

zone_results = segment_analysis(
    test_results,
    "Zone_Class"
)

print(zone_results)


# ==========================================
# 13. Pricing Consistency Analysis
# ==========================================

representative_category = (
    df["Service_Category"].mode()[0]
)

representative_subcategory = (
    df.loc[
        df["Service_Category"] == representative_category,
        "Sub_Category"
    ].mode()[0]
)

median_distance = df["Distance_KM"].median()

consistency_results = []

for zone in sorted(df["Zone_Class"].unique()):

    for peak in [0, 1]:

        for weekend in [0, 1]:

            sample = pd.DataFrame([{

                "Service_Category": representative_category,

                "Sub_Category": representative_subcategory,

                "Zone_Class": zone,

                "Distance_KM": median_distance,

                "Is_Peak_Hour": peak,

                "Is_Weekend": weekend

            }])

            predicted_price = predict_hybrid(
                best_model,
                sample
            )[0]

            consistency_results.append({

                "Zone": zone,

                "Peak_Hour": peak,

                "Weekend": weekend,

                "Predicted_Price": round(
                    predicted_price, 2
                )

            })


consistency_df = pd.DataFrame(
    consistency_results
)

print("\n========== PRICING CONSISTENCY ==========")

print(consistency_df.to_string(index=False))


# ==========================================
# 14. Permutation Feature Importance
# ==========================================

def hybrid_final_mape_scorer(estimator, X_eval, y_eval):

    predictions = predict_hybrid(
        estimator,
        X_eval
    )

    return -mean_absolute_percentage_error(
        y_eval,
        predictions
    )


print("\n========== FEATURE IMPORTANCE ==========")

importance_result = permutation_importance(

    best_model,

    X_val,

    y_val,

    n_repeats=5,

    random_state=42,

    scoring=hybrid_final_mape_scorer,

    n_jobs=-1
)


importance_df = pd.DataFrame({

    "Feature": X_val.columns,

    "Importance_Mean": np.round(
        importance_result.importances_mean, 4
    ),

    "Importance_STD": np.round(
        importance_result.importances_std, 4
    )

}).sort_values(
    by="Importance_Mean",
    ascending=False
)


print(
    importance_df.to_string(index=False)
)


# ==========================================
# 15. Save Model and Reports
# ==========================================

os.makedirs("models", exist_ok=True)


MODEL_PATH = "models/hybrid_xgboost_model.joblib"


model_artifact = {

    "model_name": "XGBoost",
    "base_pipeline": best_model,
    "context_features": context_features,
    "free_km": FREE_KM,
    "rate_per_km": RATE_PER_KM,

}


joblib.dump(
    model_artifact,
    MODEL_PATH
)


# Save Reports

comparison_df = pd.DataFrame([

    {"Dataset": "Training",**train_metrics},

    {"Dataset": "Validation",**val_metrics},

    {"Dataset": "Test",**test_metrics}
])


comparison_df.to_csv(
    "reports/xgboost_evaluation.csv",
    index=False
)

test_results.to_csv(
    "reports/test_predictions.csv",
    index=False
)

category_results.to_csv(
    "reports/category_performance.csv"
)

subcategory_results.to_csv(
    "reports/subcategory_performance.csv"
)

zone_results.to_csv(
    "reports/zone_performance.csv"
)

consistency_df.to_csv(
    "reports/pricing_consistency.csv",
    index=False
)

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)


print("\n========================================")
print("XGBOOST MODEL AND REPORTS SAVED")
