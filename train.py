# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 15:43:51 2025

@author: jindr
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import ast
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and explore the dataset
df = pd.read_csv('zomato_df_final_data.csv')


# 1. Feature engineering
# Handle missing/invalid data
# Drop columns with missing values less than 4%
df = df.dropna(subset=['cost', 'lat', 'lng', 'type', 'cost_2'])

# Impute numeric columns with mean
num_cols = ['rating_number', 'votes']
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical column with mode
cat_imputer = SimpleImputer(strategy='most_frequent')
df[['rating_text']] = cat_imputer.fit_transform(df[['rating_text']])


# Cuisine diversity
# Number of cuisines
df["num_cuisines"] = df["cuisine"].apply(len)

# Multi cuisinees
df["multicuisine"] = (df["num_cuisines"] > 1).astype(int)

# Ensure cuisine column is a list
df["cuisine"] = df["cuisine"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
)

# Flatten the cuisine list into a single series
all_cuisines = pd.Series([c for sub in df["cuisine"] for c in sub])

# Get top 20 most common cuisines
top_cuisines = all_cuisines.value_counts().head(20).index

# Explode cuisines for OneHotEncoder
cuisine_expanded = df[["cuisine"]].explode("cuisine").reset_index()

# Keep only top cuisines
cuisine_expanded = cuisine_expanded[cuisine_expanded["cuisine"].isin(top_cuisines)]

# Apply OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cuisine_ohe = ohe.fit_transform(cuisine_expanded[["cuisine"]])

# Convert to DataFrame
cuisine_ohe_df = pd.DataFrame(cuisine_ohe, columns=ohe.get_feature_names_out(["cuisine"]))
cuisine_ohe_df["index"] = cuisine_expanded["index"]

# Group back to restaurant level (sum across cuisines)
cuisine_final = cuisine_ohe_df.groupby("index").sum()

# Merge with original df
df = df.join(cuisine_final, how="left").fillna(0)

# Cost bins (quartiles) 
df["cost_bin"] = pd.qcut(df["cost"], q=3, labels=["Low", "Medium", "High"])

# Rating text (binary classes for classification tasks)
rating_binary = {
    "Poor": 0,
    "Average": 0,
    "Good": 1,
    "Very Good": 1,
    "Excellent": 1
}

df["rating_class"] = df["rating_text"].map(rating_binary).astype(int)
print(df['rating_class'].value_counts())


# 2. Regression Models
# Features and targe
features1 = ['cost','votes','cuisine_Asian', 'cuisine_Australian', 'cuisine_Bakery',
            'cuisine_Bar Food','cuisine_Burger', 'cuisine_Cafe', 'cuisine_Chinese', 
            'cuisine_Coffee and Tea','cuisine_Healthy Food', 'cuisine_Indian', 
            'cuisine_Italian', 'cuisine_Japanese','cuisine_Modern Australian', 
            'cuisine_Pizza', 'cuisine_Pub Food', 'cuisine_Sandwich','cuisine_Seafood',
            'cuisine_Sushi', 'cuisine_Thai', 'cuisine_Vietnamese','num_cuisines', 'multicuisine']
X1 = df[features1]
y1 = df['rating_number']
#print(X, y)

scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
X1_scaled = pd.DataFrame(X1_scaled, columns=X1.columns, index=X1.index)

# Training & test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)

# Model A. Linear Regression
lr = LinearRegression()

# Fit model
lr.fit(X1_train, y1_train)

# Predict
y_pred_lr = lr.predict(X1_test)

# Save lr model
with open("model.pkl", "wb") as f:
    pickle.dump(lr, f)

# Evaluate
mse_lr = mean_squared_error(y1_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y1_test, y_pred_lr)
print("Model A Linear regression")
print(f"MSE: {mse_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"R²: {r2_lr:.4f}\n")

# Save metrics
metrics_dict = {
    "MSE": mse_lr,
    "RMSE": rmse_lr,
    "R2": r2_lr
}

with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics_dict, f)
    
