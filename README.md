# **Prescription Fraud Detection**

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Data Overview](#data-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Class Imbalance Handling](#class-imbalance-handling)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Feature Importance Analysis](#feature-importance-analysis)
8. [Visualizations](#visualizations)
9. [How to Use](#how-to-use)

---

## **Project Overview**
This project leverages the **Medicare Part D Spending by Drug** dataset to detect potential fraud in prescription claims using a Random Forest Classifier. The analysis focuses on feature engineering, model evaluation, and fraud trend visualizations.

---

## **Data Overview**
- **Dataset:** Medicare Part D Spending by Drug (2022)
- **Features Include:**
  - Drug Name (Brand and Generic)
  - Manufacturer
  - Spending Metrics (Total and Per Claim)
  - Beneficiary Count
  - Fraud Indicator (Binary: 1 = Fraud, 0 = Not Fraud)

---

## **Data Preprocessing**
- **Key Steps:**
  1. Cleaned and transformed monetary and percentage fields to numeric.
  2. Handled missing values by dropping or imputing.
  3. Standardized numeric features for model compatibility.

```python
# Example code snippet:
claims = claims.dropna()
numeric_cols = [...]
for col in numeric_cols:
    claims[col] = claims[col].str.replace("$", "").str.replace(",", "").astype(float)
```

---

## **Modeling**
- Algorithm: Random Forest Classifier
- Hyperparameter Tuning: GridSearchCV to optimize parameters like:
  1. n_estimators
  2.  max_depth
  3.  max_features.
 
---

## **Class Imbalance Handling**
- Addressed the imbalance in fraud vs. non-fraud cases using SMOTE (Synthetic Minority Oversampling Technique).

```python
# from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

--- 

## **Evaluation Metrics**
- Metrics used to evaluate the model:
  - Precision
  - Recall
  - Accuracy
  - F1 Score

---
 ## **Feature Importance Analysis**
Analyzed the key features contributing to fraud detection using the trained Random Forest model.

```python
# # Example: Visualizing feature importance
import matplotlib.pyplot as plt
plt.barh(feature_names, feature_importances)
plt.title("Feature Importance")
plt.show()
```
---

## **Visualizations**

1. Fraud Rates by Drug: Highlights the drugs most prone to fraud.
2. Fraud Rates by Manufacturer: Identifies fraud-prone manufacturers.
3. Temporal Trends: Explores fraud patterns over time (2021 vs. 2022).

 
