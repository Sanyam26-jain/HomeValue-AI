# 📤 GitHub Upload Guide

## Step-by-Step Instructions

### 1. Create GitHub Repository
- Go to https://github.com/new
- **Repository name:** `House-Price-Prediction`
- **Description:** `End-to-end ML project for house price prediction using regression models with a Streamlit dashboard`
- Set to **Public**
- ✅ Add README (uncheck — we already have one)
- Click **Create repository**

### 2. Initialize Local Git Repo

```bash
cd House-Price-Prediction
git init
git add .
git commit -m "feat: initial project setup with full ML pipeline"
```

### 3. Connect to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/House-Price-Prediction.git
git branch -M main
git push -u origin main
```

### 4. Day-Wise Commit Plan

**Day 1 — Setup**
```bash
git add requirements.txt src/generate_dataset.py
git commit -m "feat: project setup and synthetic dataset generator"
```

**Day 2 — Dataset & EDA**
```bash
python main.py --eda
git add data/ images/0*.png
git commit -m "data: add synthetic dataset and EDA charts"
```

**Day 3 — Feature Engineering**
```bash
git add src/features.py src/pipeline.py
git commit -m "feat: feature engineering (Age, BathsTotal, RoomsPerArea)"
```

**Day 4 — Model Training**
```bash
python src/train_models.py
git add src/train_models.py models/
git commit -m "feat: train Linear, Ridge, DecisionTree, RandomForest, GradientBoosting"
```

**Day 5 — Evaluation**
```bash
git add images/07_model_comparison.png images/08_actual_vs_predicted.png images/09_residual_analysis.png
git commit -m "eval: model comparison, actual vs predicted, residual charts"
```

**Day 6 — Dashboard**
```bash
git add dashboard.py
git commit -m "feat: streamlit dashboard with 6 pages - predict, EDA, model comparison"
```

**Day 7 — Final Polish**
```bash
git add README.md notebooks/
git commit -m "docs: complete README, notebooks, interview prep"
git push origin main
```

### 5. Recommended GitHub Tags
```
machine-learning, regression, house-price-prediction, streamlit, 
scikit-learn, random-forest, python, data-science, eda, portfolio-project
```

### 6. Screenshots to Capture for README
- [ ] Dashboard — Predict Price page with a sample prediction
- [ ] Dashboard — Correlation heatmap
- [ ] Dashboard — Model comparison bar chart
- [ ] Dashboard — Actual vs Predicted scatter
- [ ] Terminal output from `python main.py`
- [ ] GitHub repo preview (top of repo page)

### 7. .gitignore File Content
Create `.gitignore` with:
```
__pycache__/
*.pyc
.env
venv/
*.joblib
*.pkl
.DS_Store
```
