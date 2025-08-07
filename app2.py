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
        st.write("Cleaned Data Sample:")
        st.dataframe(df.head())

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
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state['data'] = (X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        st.success("Data prepared and standardized.")

    if 'data' in st.session_state:
        st.markdown("### 📊 Model Dashboard")
        X_train_scaled, X_test_scaled, y_train, y_test = st.session_state['data'][6], st.session_state['data'][7], st.session_state['data'][4], st.session_state['data'][5]
        X_train, X_test = st.session_state['data'][2], st.session_state['data'][3]

        tabs = st.tabs(["Linear", "Ridge", "Lasso", "XGBoost"])

        with tabs[0]:
            st.header("Linear Regression")
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            st.metric("R²", round(r2_score(y_test, y_pred), 4))
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 4))
            st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            st.write("Cross-Validation Score:", np.mean(cross_val_score(model, X_train_scaled, y_train, cv=5)))
            st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))

        with tabs[1]:
            st.header("Ridge Regression")
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            st.metric("R²", round(r2_score(y_test, y_pred), 4))
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 4))
            st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            st.write("Cross-Validation Score:", np.mean(cross_val_score(ridge, X_train_scaled, y_train, cv=5)))
            st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))

        with tabs[2]:
            st.header("Lasso Regression")
            lasso = Lasso(alpha=0.1)
            lasso.fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_test_scaled)
            st.metric("R²", round(r2_score(y_test, y_pred), 4))
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 4))
            st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            st.write("Cross-Validation Score:", np.mean(cross_val_score(lasso, X_train_scaled, y_train, cv=5)))
            st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))

        with tabs[3]:
            st.header("XGBoost")
            xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            st.metric("R²", round(r2_score(y_test, y_pred), 4))
            st.metric("MSE", round(mean_squared_error(y_test, y_pred), 4))
            st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            st.write("Cross-Validation Score:", np.mean(cross_val_score(xgb, X_train, y_train, cv=5)))
            importance = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            st.bar_chart(importance)
            st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))
