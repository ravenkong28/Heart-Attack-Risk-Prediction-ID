# 💓 Heart Attack Risk Prediction – Indonesia

A machine learning project to predict the risk of heart attacks among individuals in Indonesia using real-world health data. This project compares multiple models, applies hyperparameter tuning, and interprets model predictions using SHAP.

[📂 View Dataset on Kaggle](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-)

---

## 📌 Project Overview

Cardiovascular diseases are one of the leading causes of death in Indonesia. Early detection through data-driven approaches can help reduce risks and improve public health outcomes.

- **Dataset Size**: 158,355 rows × 28 features  
- **Target Variable**: `heart_attack` (1 = Yes, 0 = No)  
- **Tools Used**: Python, Google Colab, Pandas, Scikit-learn, LightGBM, CatBoost, SHAP  

---

## 🔍 Exploratory Data Analysis (EDA)

The dataset includes a wide variety of risk factors:

- **Demographic**: `age`, `gender`, `region`, `income_level`  
- **Clinical**: `hypertension`, `diabetes`, `cholesterol_level`, `obesity`, `waist_circumference`, `family_history`  
- **Lifestyle**: `smoking_status`, `alcohol_consumption`, `physical_activity`, `dietary_habits`  
- **Environment & Stress**: `air_pollution_exposure`, `stress_level`, `sleep_hours`  
- **Medical Screening**: `blood_pressure_systolic`, `blood_pressure_diastolic`, `fasting_blood_sugar`, `cholesterol_hdl`, `cholesterol_ldl`, `triglycerides`, `EKG_results`, `previous_heart_disease`, `medication_usage`, `participated_in_free_screening`

---

## ⚙️ Preprocessing

- **Missing value imputation**: Mode for `alcohol_consumption`
- **Outlier removal**: IQR method on key numeric features
- **Encoding**: Ordinal, binary, and one-hot
- **Scaling**: StandardScaler
- **Train-test split**: 80% train, 20% test (stratified)

---

## 🤖 Model Training & Evaluation

### 🔸 Before Hyperparameter Tuning

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.7271   | 0.6870    | 0.5807 | 0.6294   | 0.8002  |
| LightGBM           | 0.7282   | 0.6841    | 0.6066 | 0.6423   | 0.8091  |
| CatBoost           | 0.7301   | 0.6846    | 0.6088 | 0.6448   | 0.8114  |

### 🔹 After Hyperparameter Tuning (GridSearchCV)

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.7271   | 0.6870    | 0.5807 | 0.6294   | 0.8002  |
| LightGBM (Tuned)   | 0.7342   | 0.6873    | 0.6129 | 0.6480   | 0.8138  |
| **CatBoost (Tuned)**| **0.7346** | **0.6888** | **0.6110** | **0.6476** | **0.8139** |

### 🛠️ Best Hyperparameters

- **Logistic Regression**:
  - `C`: 1, `penalty`: l2, `solver`: lbfgs

- **LightGBM**:
  - `num_leaves`: 31, `learning_rate`: 0.1, `n_estimators`: 200

- **CatBoost**:
  - `depth`: 8, `learning_rate`: 0.1, `iterations`: 200

---

## 🧠 SHAP Feature Importance

Top 10 influential features (based on SHAP values from CatBoost):
1. `cholesterol_level`  
2. `stress_level`  
3. `physical_activity`  
4. `waist_circumference`  
5. `blood_pressure_systolic`  
6. `fasting_blood_sugar`  
7. `cholesterol_ldl`  
8. `age`  
9. `EKG_results`  
10. `smoking_status_Current`

These features were the most influential in predicting heart attack risk, as visualized using SHAP summary plots.

---

## 🧪 Simulation Example

Given a new patient's profile:

```python
sample = {
    'age': 58,
    'gender': 'Male',
    'region': 'Urban',
    'income_level': 'Middle',
    'hypertension': 1,
    'diabetes': 0,
    'cholesterol_level': 260,
    'obesity': 1,
    'waist_circumference': 102,
    'family_history': 1,
    'smoking_status': 'Current',
    'alcohol_consumption': 'Moderate',
    'physical_activity': 'Low',
    'dietary_habits': 'Unhealthy',
    'air_pollution_exposure': 'High',
    'stress_level': 'High',
    'sleep_hours': 5,
    'blood_pressure_systolic': 150,
    'blood_pressure_diastolic': 95,
    'fasting_blood_sugar': 130,
    'cholesterol_hdl': 35,
    'cholesterol_ldl': 180,
    'triglycerides': 200,
    'EKG_results': 'Abnormal',
    'previous_heart_disease': 1,
    'medication_usage': 1,
    'participated_in_free_screening': 0
}

Predicted Risk → Heart Attack = Yes (high-risk individual)

#📎 License
This project is for academic and educational purposes only.

# 🙋‍♂️ Author
Raven Kongnando Lasher
GitHub: ravenkong28
LinkedIn: [Raven Kongnando Lasher](https://www.linkedin.com/in/ravenkongl/)
