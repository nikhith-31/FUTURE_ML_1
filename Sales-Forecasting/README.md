# 📈 Sales & Demand Forecasting — Machine Learning Project

> **Predict future monthly sales using historical business data with regression-based machine learning models.**

---

## 📋 Project Overview

This project builds a complete, end-to-end **Sales & Demand Forecasting System** that helps businesses anticipate future revenue using historical store sales data. It covers the full data science pipeline — from raw data ingestion, exploratory analysis, and feature engineering, all the way to model training, evaluation, and a 30-month forward forecast.

The project is designed to be **beginner-friendly**, **well-commented**, and **portfolio-ready** — suitable for internship submissions and GitHub showcases.

---

## ❓ Problem Statement

Businesses struggle to plan inventory, staff schedules, and cash flow because they cannot accurately predict future sales. Overstocking wastes money; understocking loses customers. A reliable forecasting model allows store managers to make data-driven decisions months in advance.

---

## 🎯 Objectives

- Analyse historical monthly sales patterns across stores and product categories
- Engineer meaningful time-series features (lags, rolling statistics, seasonality indicators)
- Train and compare multiple regression models
- Select the best model based on R² score and error metrics
- Generate a 30-month future forecast with visualisations
- Deliver actionable business insights for store owners

---

## 🛠️ Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Pandas | Data loading, cleaning, transformation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualisation |
| Scikit-learn | Model building and evaluation |
| Google Colab | Notebook execution environment |
| Jupyter Notebook | Local execution alternative |

---

## 📂 Dataset Information

The project uses a **realistic synthetic monthly store sales dataset** (`dataset.csv`) covering **3 full years (2021–2023)** with 216 rows.

| Column | Description |
|---|---|
| `Date` | First day of each month (YYYY-MM-DD) |
| `Store` | Store identifier (Store_A, Store_B) |
| `Category` | Product category (Electronics, Clothing, Groceries) |
| `Sales` | Total monthly sales in USD |

**Key characteristics:**
- 36 months of data per store/category combination
- Clear seasonal peaks in November (Black Friday) and December (Christmas)
- Upward growth trend across all categories
- Realistic noise to simulate real-world variability

---

## ⚙️ Installation Steps

### Option 1 — Google Colab (Recommended)

1. Upload `Sales_Forecasting.ipynb` and `dataset.csv` to your Google Drive
2. Open the notebook in [Google Colab](https://colab.research.google.com/)
3. Run all cells top-to-bottom (`Runtime → Run all`)

> All required libraries are pre-installed in Colab. No `pip install` needed.

### Option 2 — Local Jupyter

```bash
# 1. Clone or download this repository
git clone https://github.com/your-username/Sales-Forecasting.git
cd Sales-Forecasting

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook Sales_Forecasting.ipynb
```

---

## 🔄 Project Workflow

```
Raw CSV Data
    ↓
Data Loading & Inspection
    ↓
Data Cleaning (nulls, duplicates, types)
    ↓
Exploratory Data Analysis (8 visualisations)
    ↓
Feature Engineering (12 time-series features)
    ↓
Train / Test Split (80% / 20%)
    ↓
Model Training (Linear Regression, Random Forest, Decision Tree)
    ↓
Model Evaluation (R², MAE, MSE, RMSE)
    ↓
Best Model Selection
    ↓
30-Month Future Forecast
    ↓
Business Insights & Recommendations
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| **Linear Regression** | Baseline model; fast and interpretable |
| **Random Forest Regressor** | Ensemble model; handles non-linearity and feature interactions |
| **Decision Tree Regressor** | Tree-based model; easy to visualise and explain |

The **best model** is automatically selected based on the highest **R² score** on the test set.

---

## 📊 Results

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | ~0.88 | ~3,200 | ~4,800 |
| Random Forest | ~0.96 | ~1,900 | ~2,700 |
| Decision Tree | ~0.91 | ~2,600 | ~3,900 |

> Actual values may vary slightly depending on random seed. Random Forest typically achieves the highest R².

---

## 🖼️ Screenshots

| Chart | Description |
|---|---|
| `images/sales_trend.png` | Monthly total sales over time with rolling average |
| `images/prediction.png` | Actual vs Predicted sales on test set |
| `images/feature_importance.png` | Top features ranked by importance (Random Forest) |
| `images/error_distribution.png` | Distribution of prediction residuals |

> Charts are auto-generated when you run the notebook.

---

## 🔮 Future Scope

- Incorporate external data: weather, promotions, holidays, economic indicators
- Experiment with advanced models: XGBoost, LightGBM, ARIMA, Prophet
- Build a live dashboard with Streamlit or Dash for store managers
- Deploy the model as a REST API using FastAPI or Flask
- Add SKU-level (product-level) forecasting granularity
- Implement automated retraining when new monthly data arrives

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built as part of a data science internship portfolio project.*
