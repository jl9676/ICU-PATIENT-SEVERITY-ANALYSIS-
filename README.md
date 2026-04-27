# 🏥 ICU Patient Severity Analysis

> A Data Science project analyzing the relationship between patient age and ICU severity scores using statistical methods, EDA, and machine learning classification.

---

## 📌 Project Overview

This project applies a complete data science pipeline to an ICU (Intensive Care Unit) dataset to uncover clinical insights about how patient age correlates with severity scores, diagnosis types, and patient outcomes. The analysis supports evidence-based decision-making in critical care settings.

**Course:** 21CSS303T – Data Science  
**Domain:** Healthcare Analytics  
**Institution:** SRM Institute of Science and Technology, Vadapalani Campus  

---

## 📁 Project Structure

```
icu-severity-analysis/
│
├── dataset/
│   └── icu_dataset.csv          # ICU patient dataset (auto-generated)
│
├── notebooks/
│   └── ICU_Analysis.ipynb       # Jupyter notebook version
│
├── src/
│   └── icu_analysis.py          # Main Python script (complete pipeline)
│
├── outputs/
│   ├── plot1_severity_by_age_group.png
│   ├── plot2_age_vs_severity_scatter.png
│   ├── plot3_correlation_heatmap.png
│   ├── plot4_outcome_by_severity.png
│   ├── plot5_severity_by_diagnosis.png
│   └── plot6_confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

| Feature | Type | Description |
|---|---|---|
| age | Integer | Patient age in years (18–90) |
| gender | Categorical | Male / Female |
| diagnosis | Categorical | Sepsis / Cardiac Arrest / Respiratory Failure / Trauma / Renal Failure |
| severity_score | Float | ICU severity score (0–100) |
| los_days | Integer | Length of stay in ICU (days) |
| outcome | Categorical | Survived / Deceased |

**Records:** 300 patients | **Features:** 6 | **Missing values:** handled via median imputation

---

## ⚙️ Steps Performed

1. **Data Preprocessing** – Median imputation, age grouping, severity categorization, label encoding, Z-score normalization
2. **Feature Selection** – Pearson correlation with outcome; features with |r| > 0.1 selected
3. **EDA & Visualization** – 5 plots: bar chart, scatter plot, heatmap, stacked bar, box plot
4. **Model Building** – Logistic Regression and Random Forest Classifier (80/20 split)
5. **Evaluation** – Accuracy, classification report, confusion matrix

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~72% |
| Random Forest | ~78% |

**Key Findings:**
- Severity score increases significantly with age (Pearson r ≈ 0.60)
- Elderly patients (66+) show average severity scores ~25 points higher than young patients (18–35)
- High-severity patients have a mortality rate above 60%
- Cardiac Arrest and Sepsis show the highest median severity scores

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/icu-severity-analysis.git
cd icu-severity-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python src/icu_analysis.py

# Outputs (plots + metrics) are saved to outputs/
```

---

## 🔭 Future Work

- Integrate real ICU datasets (MIMIC-III / eICU)
- Apply XGBoost and LSTM for temporal severity prediction
- Add SHAP explainability for clinical interpretability
- Deploy as a web dashboard using Streamlit

---

## 📚 References

- Goldberger, A. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- Scikit-learn Documentation: https://scikit-learn.org
- Seaborn Documentation: https://seaborn.pydata.org
- IEA World Energy Outlook 2023: https://www.iea.org
