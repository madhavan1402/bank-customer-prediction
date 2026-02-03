# ==========================================
# Bank Customer Prediction - ML Project
# Author: Madhavan N
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ==========================================
# Load Dataset
# ==========================================
bank = pd.read_csv("bank.csv", sep=";")
bank_full = pd.read_csv("bank-full.csv", sep=";")

print("bank.csv Shape:", bank.shape)
print("bank-full.csv Shape:", bank_full.shape)


# ==========================================
# Dataset Info
# ==========================================
print("\nbank.csv Info:")
print(bank.info())

print("\nbank-full.csv Info:")
print(bank_full.info())


# ==========================================
# Encode Target Variable
# ==========================================
le = LabelEncoder()

bank['y'] = le.fit_transform(bank['y'])
bank_full['y'] = le.fit_transform(bank_full['y'])


# ==========================================
# One-Hot Encoding (Categorical Features)
# ==========================================
bank_encoded = pd.get_dummies(bank, drop_first=True)
bank_full_encoded = pd.get_dummies(bank_full, drop_first=True)


# ==========================================
# Split Features & Target
# ==========================================
X_bank = bank_encoded.drop('y', axis=1)
y_bank = bank_encoded['y']

X_bank_full = bank_full_encoded.drop('y', axis=1)
y_bank_full = bank_full_encoded['y']


# ==========================================
# Train-Test Split
# ==========================================
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank
)

Xbf_train, Xbf_test, ybf_train, ybf_test = train_test_split(
    X_bank_full, y_bank_full, test_size=0.2, random_state=42, stratify=y_bank_full
)


# ==========================================
# Train Decision Tree (bank.csv)
# ==========================================
dt_bank = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    random_state=42
)

dt_bank.fit(Xb_train, yb_train)


# ==========================================
# Evaluate bank.csv Model
# ==========================================
yb_pred = dt_bank.predict(Xb_test)

acc_bank = accuracy_score(yb_test, yb_pred)

print("\n===== bank.csv Results =====")
print("Accuracy:", round(acc_bank * 100, 2), "%\n")

print(classification_report(yb_test, yb_pred))


# Confusion Matrix
cm_bank = confusion_matrix(yb_test, yb_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_bank, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - bank.csv")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_bank.png")
plt.show()


# ==========================================
# Train Decision Tree (bank-full.csv)
# ==========================================
dt_bank_full = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    random_state=42
)

dt_bank_full.fit(Xbf_train, ybf_train)


# ==========================================
# Evaluate bank-full.csv Model
# ==========================================
ybf_pred = dt_bank_full.predict(Xbf_test)

acc_full = accuracy_score(ybf_test, ybf_pred)

print("\n===== bank-full.csv Results =====")
print("Accuracy:", round(acc_full * 100, 2), "%\n")

print(classification_report(ybf_test, ybf_pred))


# Confusion Matrix
cm_full = confusion_matrix(ybf_test, ybf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - bank-full.csv")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_bank_full.png")
plt.show()


# ==========================================
# Cross Validation (Stability Check)
# ==========================================
cv_scores = cross_val_score(
    dt_bank, X_bank, y_bank, cv=5, scoring='accuracy'
)

print("\nCross Validation Accuracy (bank.csv):")
print("Mean:", round(cv_scores.mean() * 100, 2), "%")
print("Std :", round(cv_scores.std() * 100, 2), "%")


# ==========================================
# Feature Importance
# ==========================================
importances = pd.Series(
    dt_bank.feature_importances_,
    index=X_bank.columns
).sort_values(ascending=False)


plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='bar')
plt.title("Top 15 Important Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# ==========================================
# Decision Tree Visualization
# ==========================================
plt.figure(figsize=(22, 12))

plot_tree(
    dt_bank,
    feature_names=X_bank.columns,
    class_names=['No', 'Yes'],
    filled=True,
    max_depth=3
)

plt.title("Decision Tree (bank.csv - Top Levels)")
plt.savefig("decision_tree_bank.png")
plt.show()


# ==========================================
# Export Rules
# ==========================================
rules = export_text(
    dt_bank,
    feature_names=list(X_bank.columns)
)

with open("decision_tree_rules.txt", "w") as f:
    f.write(rules)

print("\nDecision Tree Rules saved to decision_tree_rules.txt")


# ==========================================
# Summary
# ==========================================
print("\n========== PROJECT SUMMARY ==========")

print("bank.csv Accuracy     :", round(acc_bank * 100, 2), "%")
print("bank-full.csv Accuracy:", round(acc_full * 100, 2), "%")
print("CV Mean Accuracy      :", round(cv_scores.mean() * 100, 2), "%")

print("\nBank Customer Prediction Completed Successfully!")
