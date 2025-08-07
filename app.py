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
        
        
    if st.button("6. Train All Models and Show Dashboard"):
        X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = st.session_state['data']

        metrics = {}

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        y_pred_lr = lr_model.predict(X_test_scaled)
        metrics['Linear Regression'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'R2': r2_score(y_test, y_pred_lr),
            'CV R2': cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
        }

        # Ridge
        ridge = Ridge()
        ridge_grid = GridSearchCV(ridge, {'alpha': [0.1, 1.0, 10.0]}, cv=5)
        ridge_grid.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge_grid.predict(X_test_scaled)
        metrics['Ridge Regression'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'Best Alpha': ridge_grid.best_params_['alpha']
        }

        # Lasso
        lasso = Lasso(max_iter=10000)
        lasso_grid = GridSearchCV(lasso, {'alpha': [0.001, 0.01, 0.1, 1.0]}, cv=5)
        lasso_grid.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso_grid.predict(X_test_scaled)
        metrics['Lasso Regression'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
            'Best Alpha': lasso_grid.best_params_['alpha']
        }

        # XGBoost
        xgb = XGBRegressor()
        xgb_grid = GridSearchCV(xgb, {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }, cv=3)
        xgb_grid.fit(X_train, y_train)
        y_pred_xgb = xgb_grid.predict(X_test)
        metrics['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'Best Params': xgb_grid.best_params_
        }

        st.markdown("### 📊 Model Performance Dashboard")
        for model, result in metrics.items():
            st.subheader(model)
            for k, v in result.items():
                st.write(f"{k}: {v}")

        st.markdown("### 📈 XGBoost Feature Importances")
        importances = pd.Series(xgb_grid.best_estimator_.feature_importances_, index=X.columns)
        st.bar_chart(importances.nlargest(10))

        st.markdown("### 🔍 Predicted vs Actual Prices")
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(np.expm1(y_test), np.expm1(y_pred_xgb), alpha=0.5, label='XGBoost')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Predicted vs Actual House Prices")
        ax.legend()
        st.pyplot(fig)