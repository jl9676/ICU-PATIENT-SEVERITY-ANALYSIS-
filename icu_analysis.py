# ============================================================
# ICU Patient Severity Analysis
# Course: 21CSS303T - Data Science
# Author: [Your Name]
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)
# ─────────────────────────────────────────────
# SECTION 1: DATASET GENERATION (Synthetic ICU)
# ─────────────────────────────────────────────
np.random.seed(42)
N = 300
age = np.random.randint(18, 90, N)
# Severity score (0-100): older patients tend to be more severe
severity_score = np.clip(
    age * 0.6 + np.random.normal(0, 12, N), 10, 100
).astype(int)
gender = np.random.choice(['Male', 'Female'], N)
diagnosis = np.random.choice(
    ['Sepsis', 'Cardiac Arrest', 'Respiratory Failure',
     'Trauma', 'Renal Failure'], N,
    p=[0.25, 0.20, 0.25, 0.15, 0.15]
)
los_days = np.random.randint(1, 30, N)  # Length of Stay
# Outcome: higher severity → higher mortality probability
outcome_prob = severity_score / 120
outcome = np.array(
    ['Deceased' if np.random.rand() < p else 'Survived'
     for p in outcome_prob]
)
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'diagnosis': diagnosis,
    'severity_score': severity_score,
    'los_days': los_days,
    'outcome': outcome
})
# Inject some missing values for realism
df.loc[np.random.choice(df.index, 15, replace=False), 'severity_score'] = np.nan
df.loc[np.random.choice(df.index, 10, replace=False), 'los_days'] = np.nan

df.to_csv('dataset/icu_dataset.csv', index=False)
print(f"Dataset: ({df.shape[0]}, {df.shape[1]}) | Missing values before imputation: {df.isnull().sum().sum()}")
print(df.describe().round(2))
# ─────────────────────────────────────────────
# SECTION 2: DATA PREPROCESSING
# ─────────────────────────────────────────────
# 2a. Missing value imputation using median
df['severity_score'].fillna(df['severity_score'].median(), inplace=True)
df['los_days'].fillna(df['los_days'].median(), inplace=True)
print(f"\nMissing after imputation: {df.isnull().sum().sum()}")
# 2b. Age grouping (clinical bands)
bins = [17, 35, 50, 65, 90]
labels = ['Young (18-35)', 'Middle-aged (36-50)', 'Senior (51-65)', 'Elderly (66+)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
# 2c. Severity category
df['severity_cat'] = pd.cut(
    df['severity_score'],
    bins=[0, 33, 66, 100],
    labels=['Low', 'Moderate', 'High']
)
# 2d. Encode categorical variables for modelling
le_gender = LabelEncoder()
le_diag   = LabelEncoder()
le_out    = LabelEncoder()

df['gender_enc']    = le_gender.fit_transform(df['gender'])
df['diagnosis_enc'] = le_diag.fit_transform(df['diagnosis'])
df['outcome_enc']   = le_out.fit_transform(df['outcome'])   # Deceased=0, Survived=1
print("\nAge group distribution:")
print(df['age_group'].value_counts())
# ─────────────────────────────────────────────
# SECTION 3: FEATURE SELECTION (Pearson)
# ─────────────────────────────────────────────
num_features = ['age', 'severity_score', 'los_days', 'gender_enc', 'diagnosis_enc']
corr = df[num_features + ['outcome_enc']].corr()['outcome_enc'].drop('outcome_enc')
print("\nPearson r with outcome:")
print(corr.round(3))
selected_features = corr[abs(corr) > 0.1].index.tolist()
print(f"Selected features (|r|>0.1): {selected_features}")
# ─────────────────────────────────────────────
# SECTION 4: VISUALISATIONS
# ─────────────────────────────────────────────
PALETTE = ['#2E75B6', '#ED7D31', '#A9D18E', '#FF6B6B', '#9B59B6']
# --- Plot 1: Average Severity Score by Age Group (Bar Chart) ---
fig, ax = plt.subplots(figsize=(9, 5))
avg_sev = df.groupby('age_group', observed=True)['severity_score'].mean()
bars = ax.bar(avg_sev.index, avg_sev.values, color=PALETTE[:4], edgecolor='black', linewidth=0.7)
for bar, val in zip(bars, avg_sev.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Average ICU Severity Score by Age Group', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Age Group', fontsize=11)
ax.set_ylabel('Mean Severity Score', fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('outputs/plot1_severity_by_age_group.png', dpi=150)
plt.close()
print("Saved: plot1_severity_by_age_group.png")

# --- Plot 2: Scatter – Age vs Severity Score (coloured by Outcome) ---
fig, ax = plt.subplots(figsize=(9, 5))
colors = {'Survived': '#2E75B6', 'Deceased': '#FF4C4C'}
for outcome_val, grp in df.groupby('outcome'):
    ax.scatter(grp['age'], grp['severity_score'],
               c=colors[outcome_val], label=outcome_val,
               alpha=0.65, edgecolors='none', s=40)
# Trend line
z = np.polyfit(df['age'], df['severity_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['age'].min(), df['age'].max(), 200)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, label='Trend')
ax.set_title('Age vs Severity Score (by Outcome)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Age (years)', fontsize=11)
ax.set_ylabel('Severity Score', fontsize=11)
ax.legend(fontsize=10)
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/plot2_age_vs_severity_scatter.png', dpi=150)
plt.close()
print("Saved: plot2_age_vs_severity_scatter.png")
# --- Plot 3: Heatmap – Correlation Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[num_features + ['outcome_enc']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, ax=ax,
            annot_kws={'size': 10},
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('outputs/plot3_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: plot3_correlation_heatmap.png")
# --- Plot 4: Stacked Bar – Outcome Distribution across Severity Categories ---
fig, ax = plt.subplots(figsize=(8, 5))
ct = pd.crosstab(df['severity_cat'], df['outcome'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind='bar', stacked=True, ax=ax,
            color=['#FF6B6B', '#2E75B6'], edgecolor='black', linewidth=0.5)
ax.set_title('Outcome Distribution by Severity Category', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Severity Category', fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Outcome', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/plot4_outcome_by_severity.png', dpi=150)
plt.close()
print("Saved: plot4_outcome_by_severity.png")

# --- Plot 5: Box Plot – Severity Score by Diagnosis ---
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby('diagnosis')['severity_score'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='diagnosis', y='severity_score', order=order,
            palette=PALETTE, ax=ax, linewidth=0.8)
ax.set_title('Severity Score Distribution by Diagnosis', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Diagnosis', fontsize=11)
ax.set_ylabel('Severity Score', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/plot5_severity_by_diagnosis.png', dpi=150)
plt.close()
print("Saved: plot5_severity_by_diagnosis.png")
# ─────────────────────────────────────────────
# SECTION 5: MODEL BUILDING – CLASSIFICATION
# ─────────────────────────────────────────────
X = df[selected_features].copy()
X = X.fillna(X.median())
y = df['outcome_enc']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nLogistic Regression Accuracy : {acc_lr:.4f}")
print(f"Random Forest Accuracy       : {acc_rf:.4f}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=le_out.classes_))
# --- Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_out.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/plot6_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: plot6_confusion_matrix.png")
# ─────────────────────────────────────────────
# SECTION 6: KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("KEY INSIGHTS")
print("="*55)
elderly_sev = df[df['age_group'] == 'Elderly (66+)']['severity_score'].mean()
young_sev   = df[df['age_group'] == 'Young (18-35)']['severity_score'].mean()
print(f"1. Elderly patients average severity: {elderly_sev:.1f} vs Young: {young_sev:.1f}")

deceased_sev = df[df['outcome']=='Deceased']['severity_score'].mean()
survived_sev = df[df['outcome']=='Survived']['severity_score'].mean()
print(f"2. Mean severity – Deceased: {deceased_sev:.1f}, Survived: {survived_sev:.1f}")

high_mortality = (df[df['severity_cat']=='High']['outcome']=='Deceased').mean()*100
print(f"3. Mortality rate in High severity group: {high_mortality:.1f}%")

top_diag = df.groupby('diagnosis')['severity_score'].mean().idxmax()
print(f"4. Highest avg severity diagnosis: {top_diag}")

pearson_age_sev = df['age'].corr(df['severity_score'])
print(f"5. Pearson r (age vs severity): {pearson_age_sev:.3f}")

print(f"6. RF Model Accuracy: {acc_rf:.2%} | LR Accuracy: {acc_lr:.2%}")
print("\nDone – all plots saved to outputs/")
