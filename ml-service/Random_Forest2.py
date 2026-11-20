# ==========================================
# 1. Import Libraries
# ==========================================

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ==========================================
# 2. Load Dataset
# ==========================================

DATA_PATH = "sany3y_synthetic_dataset_v1.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ==========================================
# 3. Separate Features and Target
# ==========================================

TARGET = "Final_Price_EGP"

categorical_features = [
    "Service_Category",
    "Sub_Category",
    "Zone_Class"
]

numerical_features = [
    "Distance_KM",
    "Is_Peak_Hour",
    "Is_Weekend"
]

expected_columns = categorical_features + numerical_features
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

X = df[expected_columns]
y = df[TARGET]

# Context features for ML model (Distance_KM is calculated deterministically)
context_features = categorical_features + ["Is_Peak_Hour", "Is_Weekend"]


# ==========================================
# 4. Split Dataset (70% Train / 15% Val / 15% Test)
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ==========================================
# 5. Hybrid Distance Rule & Preprocessing
# ==========================================

FREE_KM = 1.5
RATE_PER_KM = 5.0

def calc_distance_fee(distances):
    return np.maximum(0.0, np.asarray(distances, dtype=float) - FREE_KM) * RATE_PER_KM

def predict_hybrid(model, X_data):
    base_price = model.predict(X_data[context_features])
    return base_price + calc_distance_fee(X_data["Distance_KM"])

# Train ML model on base service price (total price minus distance fee)
y_train_base = y_train - calc_distance_fee(X_train["Distance_KM"])

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1))
    ]
)


# ==========================================
# 6. Hyperparameter Tuning (Expanded Grid)
# ==========================================

param_grid = {
    "model__n_estimators": [200, 250, 300, 350],
    "model__max_depth": [20, 25, 30, None],
    "model__min_samples_split": [2, 5, 8, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}

cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=-1,
    verbose=1
)

print("\nStarting Grid Search...")
grid_search.fit(X_train[context_features], y_train_base)

best_model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV Base MAPE: {-grid_search.best_score_ * 100:.2f}%")


# ==========================================
# 7. Evaluate Training, Validation and Test
# ==========================================

def evaluate_model(model, X_data, y_true, dataset_name):
    y_pred = predict_hybrid(model, X_data)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} Results")
    print("-" * 35)
    print(f"MAE:  {mae:.2f} EGP")
    print(f"RMSE: {rmse:.2f} EGP")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    return y_pred

print("\n========== MODEL EVALUATION ==========")
y_train_pred = evaluate_model(best_model, X_train, y_train, "Training")
y_val_pred = evaluate_model(best_model, X_val, y_val, "Validation")
y_test_pred = evaluate_model(best_model, X_test, y_test, "Final Test")


# ==========================================
# 8. Segment Performance (Test Set)
# ==========================================

test_results = X_test.copy()
test_results["Actual_Price"] = y_test.values
test_results["Predicted_Price"] = np.round(y_test_pred, 2)
test_results["Absolute_Error"] = np.abs(test_results["Actual_Price"] - test_results["Predicted_Price"])
test_results["Absolute_Percentage_Error"] = (test_results["Absolute_Error"] / test_results["Actual_Price"]) * 100

print("\n========== PERFORMANCE BY CATEGORY ==========")
category_results = test_results.groupby("Service_Category").agg(
    Samples=("Actual_Price", "count"),
    MAE=("Absolute_Error", "mean"),
    MAPE=("Absolute_Percentage_Error", "mean")
).round(2)
print(category_results)

print("\n========== PERFORMANCE BY SUB-CATEGORY ==========")
subcategory_results = test_results.groupby("Sub_Category").agg(
    Samples=("Actual_Price", "count"),
    MAE=("Absolute_Error", "mean"),
    MAPE=("Absolute_Percentage_Error", "mean")
).round(2)
print(subcategory_results)

print("\n========== PERFORMANCE BY ZONE ==========")
zone_results = test_results.groupby("Zone_Class").agg(
    Samples=("Actual_Price", "count"),
    MAE=("Absolute_Error", "mean"),
    MAPE=("Absolute_Percentage_Error", "mean")
).round(2)
print(zone_results)


# ==========================================
# 9. Permutation Feature Importance
# ==========================================

def hybrid_mape_scorer(estimator, X_eval, y_eval):
    return -mean_absolute_percentage_error(y_eval, predict_hybrid(estimator, X_eval))

print("\n========== FEATURE IMPORTANCE ==========")
importance_result = permutation_importance(
    best_model,
    X_val,
    y_val,
    n_repeats=5,
    random_state=42,
    scoring=hybrid_mape_scorer,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance_Mean": np.round(importance_result.importances_mean, 4),
    "Importance_STD": np.round(importance_result.importances_std, 4)
}).sort_values(by="Importance_Mean", ascending=False)

print(importance_df.to_string(index=False))


# ==========================================
# 10. Save Artifacts
# ==========================================

os.makedirs("models", exist_ok=True)

test_results.to_csv("models/test_predictions.csv", index=False)
importance_df.to_csv("models/feature_importance.csv", index=False)

MODEL_PATH = "models/sany3y_hybrid_model.joblib"
model_artifact = {
    "base_pipeline": best_model,
    "context_features": context_features,
    "free_km": FREE_KM,
    "rate_per_km": RATE_PER_KM,
}
joblib.dump(model_artifact, MODEL_PATH)

print("\nSaved model & reports to 'models/' successfully.")