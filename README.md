# 🏠 House Price Prediction App

This interactive Streamlit app predicts **house prices** using machine learning models:  
**Linear Regression**, **Ridge**, **Lasso**, and **XGBoost**.  
It guides the user step-by-step through data preprocessing, feature engineering, model training, evaluation, and visualization.

---

## 🎯 Goal

To build a regression pipeline that predicts house prices from property features (e.g., size, number of bedrooms).  
The output is **continuous** (not a classification task), and the aim is to minimize prediction error using regression techniques.

---

## 📚 Theory & Models

### 📈 Linear Regression
- Predicts output using a linear combination of input features.
- Solves for weights that minimize **Mean Squared Error (MSE)**.
- Sensitive to outliers and assumes a linear relationship between features and target.

### 🔐 Ridge Regression (L2)
- Adds penalty for large weights:  
  `Loss = MSE + α * Σ(βⱼ²)`
- Helps reduce overfitting but keeps all features.

### 🧹 Lasso Regression (L1)
- Penalizes absolute value of weights:  
  `Loss = MSE + α * Σ|βⱼ|`
- Can shrink some weights to **zero**, performing automatic feature selection.

### 🌲 XGBoost (Gradient Boosting Trees)
- Builds trees sequentially, each correcting previous errors.
- Handles **non-linear patterns**, **interactions**, and **missing values**.
- Fast and accurate for tabular data.

---

## 🔄 App Workflow

### Step-by-Step Processing

| Step | Description |
|------|-------------|
| 1. Clean Data | Remove irrelevant columns: `['id', 'date', 'lat', 'long']` |
| 2. Add Feature | Create interaction feature: `sqft_living × bedrooms` |
| 3. Log Transform | Apply `log1p(price)` to reduce skew |
| 4. One-Hot Encoding | Encode `zipcode` as categorical |
| 5. Train-Test Split | Use `train_test_split` with scaling (only for linear models) |

---

## 🔢 Models Compared

| Model | Regularization | Feature Selection | Handles Non-Linearity |
|-------|----------------|-------------------|------------------------|
| Linear | ❌ | ❌ | ❌ |
| Ridge | ✅ L2 | ❌ | ❌ |
| Lasso | ✅ L1 | ✅ | ❌ |
| XGBoost | ❌ | ✅ (built-in) | ✅ |

---

## 📊 Evaluation Metrics

- **R² Score** – Goodness of fit (higher is better)
- **MSE** – Mean squared error
- **MAE** – Mean absolute error
- **Cross-Validation** – Model performance across folds

---

## 📈 Visual Outputs

- **Prediction vs Actual** (line chart)
- **Feature Importance** (XGBoost bar chart)

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

```

2. Run the Streamlit app

``` bash
streamlit run app.py

```


📬 Feedback or Suggestions?
Feel free to open issues or contribute improvements.
