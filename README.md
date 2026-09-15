# 🏠 House Price Prediction using Regression Models
## 🚀 Live Demo

👉 [Try HomeValue AI Live](https://homevalue-ai-sanyam.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

An **end-to-end Machine Learning project** that predicts house sale prices using multiple regression models. Features a fully interactive **Streamlit dashboard** with price predictor, EDA explorer, model comparison, and feature insights.

---

## 📌 Problem Statement

Pricing a property accurately is challenging — too high and it stays unsold; too low and value is lost. This project builds an **Automated Valuation Model (AVM)** that learns from historical housing data and predicts fair market prices from property features like area, quality, location, and age.

---

## 🌍 Industry Relevance

| Industry | Use Case |
|----------|----------|
| Real Estate Portals | Instant listing price suggestions |
| Banks & NBFCs | Mortgage underwriting / collateral valuation |
| iBuyers | Automated property offers |
| Homeowners | Renovation ROI calculation |
| Investors | Portfolio valuation |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn (LinearRegression, Ridge, DecisionTree, RandomForest, GradientBoosting) |
| Dashboard | Streamlit |
| Model Saving | Joblib |
| Data Format | CSV |

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── houses.csv              ← Auto-generated synthetic dataset (1500 rows)
│
├── src/
│   ├── generate_dataset.py     ← Synthetic data generator
│   ├── features.py             ← Feature engineering (Age, BathsTotal, etc.)
│   ├── pipeline.py             ← Scikit-learn ColumnTransformer preprocessing
│   ├── train_models.py         ← Model training + comparison + saving
│   └── eda_visualize.py        ← EDA charts saved to images/
│
├── models/
│   ├── best_model.joblib       ← Best trained pipeline
│   ├── all_models.joblib       ← All trained model pipelines
│   └── results.json            ← Evaluation metrics
│
├── images/
│   ├── 01_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_vs_area.png
│   ├── 04_price_by_neighborhood.png
│   ├── 05_price_by_quality.png
│   ├── 06_age_vs_price.png
│   ├── 07_model_comparison.png
│   ├── 08_actual_vs_predicted.png
│   └── 09_residual_analysis.png
│
├── notebooks/
│   └── exploration.ipynb       ← Jupyter notebook for step-by-step walkthrough
│
├── outputs/                    ← Prediction CSVs / reports
│
├── dashboard.py                ← 🚀 Streamlit dashboard (main UI)
├── main.py                     ← CLI runner (full pipeline)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Full Pipeline (Train Models + Generate Charts)

```bash
python main.py
```

### 5. Launch Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser.

### 6. Interactive CLI Predictor

```bash
python main.py --predict
```

---

## 🤖 Models Used

| Model | Description | Best For |
|-------|-------------|----------|
| Linear Regression | Baseline parametric model | Interpretability |
| Ridge Regression | Regularized, handles multicollinearity | Correlated features |
| Decision Tree | Rule-based non-linear model | Explainability |
| **Random Forest** | **Ensemble of trees (bagging)** | **Best accuracy + robustness** |
| Gradient Boosting | Sequential trees (boosting) | High accuracy |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average ₹ error |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R²** | R-Squared — % variance explained (0–1, higher is better) |

---

## 🔧 Feature Engineering

New derived features created from raw data:

| Feature | Formula | Why |
|---------|---------|-----|
| `Age` | 2024 − YearBuilt | House depreciation |
| `RemodAge` | 2024 − YearRemodAdd | Renovation recency |
| `BathsTotal` | FullBath + 0.5×HalfBath | Composite bath count |
| `RoomsPerArea` | TotRooms / GrLivArea | Density metric |
| `GarageQualCars` | GarageCars × (GarageArea/200) | Garage quality |
| `TotalSF` | Basement + 1st + 2nd floor | Total usable area |
| `WasRemodeled` | 1 if remodeled after build | Binary flag |

---

## 🖥️ Dashboard Features

| Page | Content |
|------|---------|
| 🏠 **Predict Price** | Enter property details → get ML price estimate + what-if analysis |
| 📊 **Data Explorer** | Price distribution, neighborhood box plots, correlation heatmap, scatter plots |
| 📈 **Model Performance** | RMSE/R² comparison, actual vs predicted, residuals |
| 🔍 **Feature Insights** | Feature importance, price by category, age effects |
| 📋 **Dataset Preview** | Filterable table with search |
| ℹ️ **About** | Project overview, architecture, industry use cases |

---

## 📈 Sample Results

```
Model                   Test RMSE          R²
────────────────────────────────────────────
Linear Regression      ₹1,850,000       0.81
Ridge Regression       ₹1,820,000       0.82
Decision Tree          ₹1,600,000       0.86
Random Forest          ₹1,200,000       0.92
Gradient Boosting      ₹1,280,000       0.91
```

---

## 🗂️ Dataset

- **Source:** Synthetically generated (1,500 rows) with realistic distributions
- **Features:** 25 property attributes (area, quality, rooms, location, age, etc.)
- **Target:** `SalePrice` in Indian Rupees (₹)
- **Missing Values:** ~5% in TotalBsmtSF, GarageCars, GarageArea (handled by imputer)

---

## 🎓 Learning Outcomes

- ✅ Supervised learning — regression problem framing
- ✅ Feature engineering — derived features, domain knowledge
- ✅ Preprocessing pipelines — ColumnTransformer, SimpleImputer, OneHotEncoder
- ✅ Model comparison — cross-validation, hold-out evaluation
- ✅ Visualization — EDA, residual analysis, feature importance
- ✅ Model persistence — joblib save/load
- ✅ Dashboard development — Streamlit

---

## 💼 Interview Preparation

**Q: What is this project about?**  
Built an end-to-end ML regression system that predicts residential house prices from property features. Covers full pipeline: data generation → EDA → feature engineering → model training → evaluation → Streamlit deployment.

**Q: Why regression and not classification?**  
Price is a continuous numerical value (₹), not a category. Regression directly predicts this value.

**Q: Which model performed best?**  
Random Forest and Gradient Boosting had the best R² (~0.92). Random Forest was chosen for its balance of accuracy, speed, and feature importance interpretability.

**Q: How did you handle missing values?**  
Used `SimpleImputer(strategy="median")` for numeric columns and `most_frequent` for categoricals inside a scikit-learn Pipeline.

**Q: How can this be improved?**  
- Use real Kaggle House Prices dataset
- Add XGBoost / LightGBM
- Hyperparameter tuning with Optuna
- SHAP explainability
- REST API with FastAPI
- Docker containerization

---

## 🏷️ Tags

`machine-learning` `regression` `house-price-prediction` `streamlit` `scikit-learn` `random-forest` `gradient-boosting` `data-science` `python` `eda` `feature-engineering` `portfolio-project`

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.

---

> Built with ❤️ as a student portfolio project demonstrating real-world ML workflows.
