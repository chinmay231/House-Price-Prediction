import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("🏠 House Price Prediction App")

uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Step-by-step Processing")

    if st.button("1. Clean Data"):
        st.write("Drop irrelevant or non-numeric columns. Also explicitly remove 'date' to prevent float conversion errors.")
        drop_cols = ['id', 'date', 'lat', 'long']
        for col in drop_cols:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        df['zipcode'] = df['zipcode'].astype(str)
        st.write("Cleaned Data Sample:", df.head())

    if st.button("2. Add Interaction Feature"):
        st.write("Add new feature: sqft_living × bedrooms to capture size per room.")
        df['sqft_bedrooms'] = df['sqft_living'] * df['bedrooms']
        st.write(df[['sqft_bedrooms']].head())

    if st.button("3. Log-transform Target"):
        st.write("Log-transform the skewed price to reduce the effect of outliers.")
        df['price'] = np.log1p(df['price'])
        st.write(df['price'].head())

    if st.button("4. One-Hot Encode Zipcode"):
        st.write("Convert categorical zipcodes into numeric columns using one-hot encoding.")
        df = pd.get_dummies(df, columns=['zipcode'], drop_first=True)
        st.write(df.head())

    if st.button("5. Split & Standardize"):
        st.write("Split dataset into training/testing and scale features (for linear models only).")
        X = df.drop('price', axis=1)
        y = df['price']
        # Ensure all X values are numeric
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state['data'] = (X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        st.success("Data prepared and standardized.")

    # Remainder of code remains unchanged...
