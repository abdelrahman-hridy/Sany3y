# Sany3y AI — Smart Service Pricing Engine

An end-to-end Machine Learning pricing engine built for the **Sany3y** home-services marketplace. The model predicts fair labor prices (in EGP, excluding spare parts/materials) for on-demand maintenance requests and provides a reliable baseline for the platform's technician bidding system.

## 1. Key Highlights & Results

* **Model Explanatory Power ($R^2$ Score):** `0.9709` on the unseen test set (explains **97.09%** of price variance).
* **Relative Pricing Accuracy:** **93.22%** (`MAPE = 6.78%`, `MAE = 22.63 EGP`, `RMSE = 31.08 EGP`).
* **Behavioral Guarantees:** **100.00%** distance monotonicity (zero price drops as distance increases) and **100%** adherence to zone and surge-time pricing rules across all 25 services.
* **Zero Data Leakage:** Built using Scikit-Learn `Pipeline` and `ColumnTransformer` fitted strictly on the training split.

## 2. Project Structure

```text
random_forest/
├── generate_data.py                  # Synthetic dataset generator (Dataset V1)
├── sany3y_synthetic_dataset_v1.csv   # 10,000 labeled service requests
├── Random_Forest2.py                 # End-to-end preprocessing, training, GridSearchCV & evaluation
├── predict_interactive.py            # Interactive CLI simulator for live price testing
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── models/
    ├── sany3y_hybrid_model.joblib    # Serialized hybrid model bundle (pipeline + distance config)
    ├── test_predictions.csv          # Detailed predictions & error metrics on the 1,500 test rows
    └── feature_importance.csv        # Permutation feature importance rankings
```

## 3. Dataset & Feature Schema

The dataset (`sany3y_synthetic_dataset_v1.csv`) contains **10,000 records** distributed across **5 main categories**, **25 sub-categories**, and **3 socio-economic zone classes**:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Service_Category` | Categorical | Main trade (`Carpentry`, `Electricity`, `HVAC`, `Painting`, `Plumbing`) |
| `Sub_Category` | Categorical | Specific task (25 distinct maintenance services) |
| `Zone_Class` | Categorical | Geographic pricing tier (`Class_A`, `Class_B`, `Class_C`) |
| `Distance_KM` | Numerical | Distance between technician and customer in kilometers |
| `Is_Peak_Hour` | Binary (`0/1`) | Whether the request is placed during peak demand hours |
| `Is_Weekend` | Binary (`0/1`) | Whether the request falls on a weekend |
| `Final_Price_EGP` | **Target** | Total labor + transportation fee in Egyptian Pounds (EGP) |

## 4. Preprocessing & Feature Encoding Pipeline

All feature transformations are encapsulated inside a Scikit-Learn `ColumnTransformer` and `Pipeline` to guarantee identical preprocessing during training and production inference while preventing data leakage:

1. **Categorical Encoding (`OneHotEncoder`):**
   * Applied to `Service_Category` (5 categories), `Sub_Category` (25 categories), and `Zone_Class` (3 categories) with `handle_unknown="ignore"`.
   * Expands the **3 categorical columns** into **33 binary indicator features** ($5 + 25 + 3 = 33$).
2. **Binary Feature Passthrough (`remainder="passthrough"`):**
   * `Is_Peak_Hour` and `Is_Weekend` are already clean binary indicators (`0` or `1`) and pass directly into the model without modification.
   * Combined with the encoded columns, the Random Forest estimator receives a **35-feature matrix** ($33 + 2 = 35$).
3. **Why No Feature Scaling?**
   * Decision trees and Random Forests split data based on threshold inequalities rather than Euclidean distances or gradient magnitudes, making feature scaling (`StandardScaler` / `MinMaxScaler`) unnecessary.
4. **Target Isolation (Distance Decoupling):**
   * During training, the deterministic transportation fee is subtracted from `Final_Price_EGP` so the Random Forest learns pure contextual labor pricing without tree-splitting artifacts across continuous distances.

## 5. Hybrid Pricing Architecture (Engineering Case Study)

During sensitivity testing across **975 scenarios** (25 services $\times$ 39 distances from `1.0 km` to `20.0 km`), a pure Random Forest trained directly on `Distance_KM` achieved good average error but exhibited step-function fluctuations (local non-monotonicity of **67.05%**, with occasional price drops up to `-41.42 EGP` as distance increased).

To solve this while preserving high predictive accuracy, we implemented a **Hybrid Pricing Architecture**:

1. **Contextual ML Estimator (Random Forest):** Predicts the clean base labor price using `Service_Category`, `Sub_Category`, `Zone_Class`, `Is_Peak_Hour`, and `Is_Weekend`.
2. **Deterministic Distance Engine:** Calculates transportation fees via an exact business rule (first `1.5 km` free, then `5.0 EGP/km`):

$$\text{Distance Fee} = \max(0, d - 1.5) \times 5$$

$$\text{Final Predicted Price} = \hat{y}_{\text{base}}(\text{context}) + \text{Distance Fee}$$

### Pure RF vs. Hybrid RF Behavioral Comparison

| Behavioral Metric (Across All 25 Services) | Pure Random Forest | Hybrid Random Forest |
| :--- | :---: | :---: |
| **Distance Monotonicity (950 steps)** | 67.05% (313 drops) | **100.00% (0 drops)** |
| **Worst Price Drop on Distance Increase** | -41.42 EGP | **0.00 EGP** |
| **Zone Pricing Order (`Class_A > Class_B > Class_C`)** | 24 / 25 (96.0%) | **25 / 25 (100.0%)** |
| **Surge Time Uplift (`Peak` & `Weekend`)** | 22 / 25 (88.0%) | **25 / 25 (100.0%)** |

## 6. Training & Evaluation Results

### Data Split & Hyperparameter Tuning

* **Split Ratio:** 70% Training (`7,000` rows) / 15% Validation (`1,500` rows) / 15% Testing (`1,500` rows).
* **Optimization:** 3-Fold `KFold` `GridSearchCV` evaluating **576 hyperparameter combinations** (1,728 total fits).
* **Best Parameters:** `n_estimators=200`, `max_depth=30`, `min_samples_split=10`, `min_samples_leaf=2`, `max_features=None`.

### Generalization Check (Train vs. Validation vs. Test)

| Dataset Split | Samples | MAE (EGP) | RMSE (EGP) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training (70%)** | 7,000 | 21.37 | 29.99 | 6.40% | 0.9708 |
| **Validation (15%)** | 1,500 | 22.71 | 32.09 | 6.67% | 0.9686 |
| **Final Test (15%)** | **1,500** | **22.63** | **31.08** | **6.78%** | **0.9709** |

*The minimal gap between training (`6.40%` MAPE) and unseen testing (`6.78%` MAPE) confirms strong generalization with no overfitting.*

### Performance Breakdown by Service Category (Test Set)

| Service Category | Test Samples | MAE (EGP) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Carpentry** | 314 | 15.93 | 6.66% |
| **Electricity** | 307 | 16.52 | 7.20% |
| **HVAC** | 307 | 30.71 | 6.15% |
| **Painting** | 279 | 34.00 | 6.95% |
| **Plumbing** | 293 | 16.93 | 6.97% |

### Performance Breakdown by Zone Class (Test Set)

| Zone Class | Test Samples | MAE (EGP) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Class_A** | 352 | 25.71 | 6.93% |
| **Class_B** | 706 | 22.52 | 6.81% |
| **Class_C** | 442 | 20.34 | 6.62% |

### Permutation Feature Importance (Validation Set)

| Rank | Feature | Importance Mean (MAPE Impact) | Std Dev |
| :--- | :--- | :---: | :---: |
| 1 | `Sub_Category` | 0.3509 | $\pm 0.0095$ |
| 2 | `Service_Category` | 0.1825 | $\pm 0.0031$ |
| 3 | `Distance_KM` | 0.0735 | $\pm 0.0011$ |
| 4 | `Zone_Class` | 0.0471 | $\pm 0.0012$ |
| 5 | `Is_Peak_Hour` | 0.0361 | $\pm 0.0007$ |
| 6 | `Is_Weekend` | 0.0171 | $\pm 0.0005$ |

## 7. Setup & Usage

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 1: Generate Dataset (Optional if CSV already exists)

```bash
python generate_data.py
```

### Step 2: Train & Evaluate the Model

Runs the full preprocessing, encoding, grid search, evaluation, and saves the trained artifact to `models/sany3y_hybrid_model.joblib`:

```bash
python Random_Forest2.py
```

### Step 3: Run the Interactive Pricing Simulator

Test custom service requests via an interactive terminal menu:

```bash
python predict_interactive.py
```



## 8. Future Roadmap (Production Phase)

1. **Real-World Feedback Loop:** Log actual accepted bids from completed orders in the Sany3y production database to periodically retrain the model on real Egyptian market dynamics.
2. **Dynamic Zone Calibration:** Adjust zone multipliers dynamically based on real-time supply and demand per district.
3. **Model Benchmarking:** Benchmark against Gradient Boosting architectures (`LightGBM` / `XGBoost`) once real historical transaction logs reach scale.