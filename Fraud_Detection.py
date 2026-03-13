#!/usr/bin/env python
# coding: utf-8

# # Fraud Transaction Detection

# ## Objective
# Build a system that can classify if a transaction is fraudulent or not using simulated transaction data.

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler
import joblib

import warnings
warnings.filterwarnings('ignore')
print("Libraries loaded successfully.")


# ## Load Dataset (from daily .pkl files)

# In[ ]:


# Load all daily transaction .pkl files and combine
data_dir = 'dataset/data'
pkl_files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))

print(f"Found {len(pkl_files)} daily data files.")

# Load first 30 days for a manageable dataset (use all files if you have enough RAM)
df_list = [pd.read_pickle(f) for f in pkl_files[:30]]
df = pd.concat(df_list, ignore_index=True)

print(f"Combined dataset shape: {df.shape}")
df.head()


# ## Exploratory Data Analysis

# In[ ]:


# Basic info
print(df.dtypes)
print("\nFraud distribution:")
print(df['TX_FRAUD'].value_counts())

fraud_pct = df['TX_FRAUD'].mean() * 100
print(f"\nFraud rate: {fraud_pct:.2f}%")

# Plot fraud distribution
plt.figure(figsize=(6, 4))
df['TX_FRAUD'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.xticks([0, 1], ['Legitimate', 'Fraudulent'], rotation=0)
plt.title('Transaction Class Distribution')
plt.ylabel('Count')
plt.tight_layout()
# plt.show()


# ## Feature Engineering & Preprocessing

# In[ ]:


# Parse datetime
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek

# Terminal-level feature: avg transaction amount per terminal
terminal_stats = df.groupby('TERMINAL_ID')['TX_AMOUNT'].agg(['mean', 'count']).reset_index()
terminal_stats.columns = ['TERMINAL_ID', 'TERMINAL_AVG_AMOUNT', 'TERMINAL_TX_COUNT']
df = df.merge(terminal_stats, on='TERMINAL_ID', how='left')

# Customer-level feature: avg transaction amount per customer
customer_stats = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].agg(['mean', 'count']).reset_index()
customer_stats.columns = ['CUSTOMER_ID', 'CUSTOMER_AVG_AMOUNT', 'CUSTOMER_TX_COUNT']
df = df.merge(customer_stats, on='CUSTOMER_ID', how='left')

# Select features
features = ['TX_AMOUNT', 'TX_HOUR', 'TX_DAY_OF_WEEK',
            'TERMINAL_AVG_AMOUNT', 'TERMINAL_TX_COUNT',
            'CUSTOMER_AVG_AMOUNT', 'CUSTOMER_TX_COUNT']

X = df[features]
y = df['TX_FRAUD']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")


# ## Model Building (Gradient Boosting / Random Forest)

# In[ ]:


# Use Random Forest (fast and robust)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
print("Model trained!")


# ## Training & Evaluation

# In[ ]:


y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
# plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve')
# plt.show()


# ## Saving Model

# In[ ]:


joblib.dump(rf, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'fraud_scaler.pkl')
print("Model and scaler saved.")

