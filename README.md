# 🏦 Personal Loan Acceptance Prediction

A machine learning project to predict which bank customers are likely to accept a personal loan offer — using the UCI Bank Marketing Dataset with 45,211 customer records.

---

## 📋 Problem Statement

A Portuguese bank runs telemarketing campaigns to offer personal loans to existing customers. Calling every customer is expensive and inefficient. This project builds a classification model to **predict which customers are most likely to accept** the loan offer — so the bank can focus its marketing efforts on the right people and reduce wasted calls.

**Target Variable:** `y` → Did the customer accept the loan? (`yes` = 1, `no` = 0)

---

## 🗂️ Dataset

**File:** `bank-full.csv`

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

**Size:** 45,211 rows × 17 columns

| Feature | Type | Description |
|---|---|---|
| age | Numerical | Age of the customer |
| job | Categorical | Type of job |
| marital | Categorical | Marital status |
| education | Categorical | Education level |
| default | Categorical | Has credit in default? |
| balance | Numerical | Average yearly balance (EUR) |
| housing | Categorical | Has housing loan? |
| loan | Categorical | Has personal loan? |
| contact | Categorical | Contact communication type |
| day | Numerical | Last contact day of month |
| month | Categorical | Last contact month |
| duration | Numerical | Last contact duration (seconds) |
| campaign | Numerical | Number of contacts this campaign |
| pdays | Numerical | Days since last contact (previous campaign) |
| previous | Numerical | Contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign |
| **y** | **Binary** | **TARGET — Accepted loan? (yes/no)** |

---

## 🛠️ Tools & Libraries

```
Python 3
├── pandas          — data loading and manipulation
├── numpy           — numerical operations
├── matplotlib      — plotting and charts
├── seaborn         — heatmaps and styled plots
└── scikit-learn    — encoding, models, evaluation
```

---

## ⚙️ Project Pipeline

### Step 1 — Data Exploration
- Dataset: **45,211 customers**, 17 features
- Class imbalance: **~88% declined, ~12% accepted**
- No missing values found
- Explored acceptance rates by age, job, marital status, education, and balance

### Step 2 — EDA Visualizations
- Target distribution (count + pie chart)
- Age distribution by acceptance + acceptance rate by age group
- Acceptance rate by job type and marital status
- Acceptance rate by education level
- Balance distribution by acceptance
- Feature correlation heatmap

### Step 3 — Data Cleaning & Encoding

**Unknown value handling:**
- Replaced `'unknown'` entries with column mode values

**Label Encoding** → used for binary columns
```
sex      → female = 0,  male = 1
smoker   → no = 0,      yes  = 1
default  → no = 0,      yes  = 1
housing  → no = 0,      yes  = 1
loan     → no = 0,      yes  = 1
```

**One-Hot Encoding** → used for multi-class columns
```
job, marital, education, contact, month, poutcome
→ each split into multiple binary columns
```

### Step 4 — Train / Test Split
- Split ratio: **80% training / 20% testing**
- Stratified split to preserve the 12% acceptance ratio
- `StandardScaler` applied for Logistic Regression
- `class_weight='balanced'` used on both models to handle imbalance

### Step 5 — Model Training
Two classifiers were trained and compared:

| Model | Key Settings |
|---|---|
| Logistic Regression | max_iter=1000, balanced weights |
| Decision Tree | max_depth=5, balanced weights |

---

## 📊 Results

### Model Comparison

| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| **Logistic Regression** | **84.00%** | **90.41%** | **53.00%** |
| Decision Tree | 77.00% | 86.42% | 46.00% |

### Classification Report — Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Declined (0) | 0.97 | 0.85 | 0.90 | 7,985 |
| Accepted (1) | 0.40 | 0.79 | 0.53 | 1,058 |
| **Accuracy** | | | **0.84** | **9,043** |

### Classification Report — Decision Tree

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Declined (0) | 0.97 | 0.76 | 0.85 | 7,985 |
| Accepted (1) | 0.32 | 0.84 | 0.46 | 1,058 |
| **Accuracy** | | | **0.77** | **9,043** |

### 5-Fold Cross Validation (AUC-ROC)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 90.59% | 90.04% | 90.32% | 90.20% | 90.87% | **90.41%** | ±0.30% |
| Decision Tree | 87.51% | 85.23% | 86.09% | 85.79% | 87.49% | **86.42%** | ±0.92% |

---

## 🏆 Best Model — Logistic Regression

| Why Logistic Regression wins | Detail |
|---|---|
| Higher AUC-ROC | 90.41% vs 86.42% — better at ranking loan acceptors |
| More consistent | ±0.30% std vs ±0.92% — very stable across folds |
| Better recall on accepted | Catches 79% of real acceptors |
| Higher F1 on accepted class | 0.53 vs 0.46 |

> **Key insight:** Logistic Regression correctly identifies 79% of customers who will accept the loan — meaning the bank can save significant marketing budget by calling only flagged customers.

---

## 📌 Customer Groups Most Likely to Accept

### By Job Type (Top 5)
| Job | Acceptance Rate |
|---|---|
| Student | **28.7%** |
| Retired | **22.8%** |
| Unemployed | 15.5% |
| Management | 13.8% |
| Admin | 12.2% |

### By Marital Status
| Marital Status | Acceptance Rate |
|---|---|
| Single | **14.9%** |
| Divorced | 11.9% |
| Married | 10.1% |

### By Education Level
| Education | Acceptance Rate |
|---|---|
| Tertiary | **15.0%** |
| Secondary | 10.6% |
| Primary | 8.6% |

### By Age Group
| Age Group | Acceptance Rate |
|---|---|
| 60+ | **42.3%** |
| Under 30 | **16.3%** |
| 30–40 | 10.2% |
| 50–60 | 10.1% |
| 40–50 | 9.1% |

---

## 💡 Key Business Insights

**1. Age 60+ is the highest-converting group (42.3%)**
Retired and elderly customers are far more likely to accept — nearly 1 in 2 customers in this group says yes. This should be the bank's primary target segment.

**2. Students accept at a surprisingly high rate (28.7%)**
Young students are the top-converting job category, likely because they have fewer financial commitments and are open to new financial products.

**3. Retired customers are the 2nd best job segment (22.8%)**
Retired customers have stable income and more time to engage — they respond well to direct contact campaigns.

**4. Single customers accept more than married ones (14.9% vs 10.1%)**
Single customers have fewer financial obligations and are more open to taking on new loans.

**5. Tertiary education leads to more acceptances (15.0%)**
Higher-educated customers understand financial products better and are more willing to engage with loan offers.

**6. Logistic Regression is more consistent than Decision Tree**
With a cross-validation std of only ±0.30%, Logistic Regression is far more stable and reliable for deployment in a real campaign.

---

## 📁 Project Structure

```
personal-loan-prediction/
│
├── bank-full.csv                        ← Full dataset (45,211 rows)
├── personal_loan_prediction_colab.py    ← Full Colab code (22 cells)
├── README.md                            ← This file
│
└── charts/
    ├── target_distribution.png          ← Loan acceptance pie + bar
    ├── age_analysis.png                 ← Age distribution by acceptance
    ├── job_marital_analysis.png         ← Job & marital acceptance rates
    ├── education_balance_analysis.png   ← Education & balance charts
    ├── correlation_heatmap.png          ← Feature correlation heatmap
    ├── confusion_matrices.png           ← Both models side by side
    ├── roc_curves.png                   ← ROC curves comparison
    ├── decision_tree.png                ← Decision tree visualization
    └── feature_importance.png           ← Top features bar chart
```

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Paste cells from `personal_loan_prediction_colab.py` one by one
4. Run **Cell 1** first (imports)
5. Upload `bank-full.csv` when prompted in **Cell 2**
6. Run all remaining cells in order

---
