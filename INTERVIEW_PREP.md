# 🎯 Interview Preparation — House Price Prediction

## Core ML Questions

**Q1. Explain your project in 2 minutes.**
> "I built an end-to-end house price prediction system using supervised regression ML. 
> It takes property features like area, quality, location, and age as input, and outputs 
> a predicted market value. The project covers the full pipeline — data generation, EDA, 
> feature engineering, training 5 regression models, comparing them on RMSE/R², and 
> serving predictions via a Streamlit dashboard. The best model (Random Forest) achieves 
> R² ~0.92 on held-out test data."

---

**Q2. Why is this a regression and not classification problem?**
> "The target (SalePrice) is a continuous numerical value — we're predicting an exact 
> price in rupees. Classification would categorize into buckets (low/mid/high), which 
> loses precision. Regression directly minimizes price error."

---

**Q3. What evaluation metrics did you use and why?**
> - **MAE** — easy to interpret (average ₹ error), robust to outliers  
> - **RMSE** — penalizes large errors more (good for business, where a ₹50L error is worse than 5× ₹10L errors)  
> - **R²** — explains what % of price variance the model captures; 0.92 means the model explains 92% of variance

---

**Q4. What feature engineering did you do?**
> Added domain-driven derived features:
> - `Age` = 2024 − YearBuilt (depreciation effect)
> - `BathsTotal` = FullBath + 0.5×HalfBath (composite)
> - `RoomsPerArea` = rooms/area (density)
> - `WasRemodeled` (binary flag)
> These improved model performance because raw year/bathroom columns hide these relationships.

---

**Q5. How did you handle missing values?**
> Used scikit-learn's `SimpleImputer` inside a `Pipeline`:
> - Numeric columns → median imputation (robust to outliers)
> - Categorical columns → most_frequent imputation
> This ensures the same strategy is applied consistently at both training and inference time.

---

**Q6. Why did Random Forest outperform Linear Regression?**
> Linear Regression assumes a linear relationship between features and price. Housing 
> prices have non-linear interactions (e.g., quality × location, area × age). Random 
> Forest handles these automatically through deep trees and averaging across 200+ trees, 
> reducing variance without overfitting.

---

**Q7. What is R² score? What does R²=0.92 mean?**
> R² measures the proportion of variance in the target explained by the model. 
> R²=0.92 means the model explains 92% of the variability in house prices. 
> The remaining 8% is noise or unmeasured factors.

---

**Q8. How did you prevent overfitting in Random Forest?**
> - `max_depth=12` limits tree depth
> - `min_samples_leaf=5` prevents fitting on single samples  
> - 5-fold cross-validation to detect and measure overfitting
> - Train/test split (80/20) for unbiased evaluation

---

**Q9. How would you deploy this to production?**
> 1. Wrap the model in a FastAPI REST endpoint (`/predict`)
> 2. Containerize with Docker
> 3. Deploy on AWS/GCP/Azure or Render
> 4. Add input validation (Pydantic models)
> 5. Monitor RMSE on new incoming data weekly
> 6. Retrain if drift is detected (feature distributions shift)

---

**Q10. How can this project be improved further?**
> - Use real Kaggle House Prices dataset (Iowa, Ames)
> - Add XGBoost / LightGBM (typically 5–10% better)
> - Optuna hyperparameter tuning
> - SHAP values for feature explanations per prediction
> - Geospatial features (distance to metro, schools, hospitals)
> - Time-series correction for market inflation

---

## Data Science / Analyst Questions

**Q11. What is multicollinearity and how does it affect linear regression?**
> When two features are highly correlated (e.g., GrLivArea and TotalSF), linear 
> regression coefficients become unstable. Ridge regression handles this by adding 
> an L2 penalty that shrinks correlated coefficients.

**Q12. Why did you log-transform the price?**
> Raw prices are right-skewed (few very expensive houses pull the mean up). 
> Log-transforming makes the distribution more normal, which helps linear models 
> fit better and reduces the influence of extreme outliers on RMSE.

**Q13. What does the correlation heatmap tell you?**
> It shows how strongly each feature correlates with SalePrice. In this dataset, 
> OverallQual (0.79), GrLivArea (0.71), TotalSF (0.75) had highest correlations — 
> confirming quality and size are the primary price drivers.
