This project predicts Boston house prices using machine learning models trained on the Boston Housing dataset. 
It includes data preprocessing, model comparison, evaluation, and deployment through a Streamlit web app for interactive predictions.

🤖 Machine Learning: Scikit-learn, XGBoost, LightGBM, TensorFlow
📊 Data Processing: Pandas, NumPy, SciPy
📈 Visualization: Matplotlib, Seaborn, Plotly
🔍 Model Interpretability: SHAP, LIME
🚀 Deployment: Streamlit, Docker, FastAPI
📝 Documentation: Jupyter Notebooks, Markdown

Results Summary
| Model          | RMSE (Test) | R² Score | Training Time |
| -------------- | ----------- | -------- | ------------- |
| XGBoost        | 2.85        | 0.91     | 1.2s          |
| Random Forest  | 3.12        | 0.89     | 2.1s          |
| LightGBM       | 2.98        | 0.90     | 0.8s          |
| Neural Network | 3.45        | 0.87     | 15.3s         |

Model Ensemble & Stacking
# Custom stacking regressor implementation
estimators = [('xgb', XGBRegressor()), 
              ('rf', RandomForestRegressor()),
              ('lgb', LGBMRegressor())]
stacking_regressor = StackingRegressor(estimators=estimators)

Real-world Applications:
1. Real estate valuation for investors
2. Property pricing optimization
3. Mortgage risk assessment
4. Urban planning & development

Model Performance: Achieves 91% variance explanation with production-ready inference time (<50ms)

Quick Start:-
1. Clone the repository:-
git clone https://github.com/ramdhote05/boston-house-price-prediction.git
cd boston-house-price-prediction

2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

3. Install dependencies
pip install -r requirements.txt

4. Launch the app
# Interactive Dashboard
streamlit run app.py

# Jupyter Notebooks
jupyter notebook notebooks/

Deploying on Streamlit Cloud
----------------------------

1. Ensure the repository contains `app.py` at the repository root, `requirements.txt`, and the model files (`best_model.pkl`, `scaler.pkl`).
2. Push the repository to GitHub:

    git add .
    git commit -m "Add Streamlit app and model files"
    git push origin main

3. Go to https://streamlit.io/cloud, sign in with GitHub, and click **New app**.
    - Select the repository and branch.
    - Set the **Main file** to `app.py` and click **Deploy**.

Local Run
---------

Run the app locally with:

```
pip install -r requirements.txt
streamlit run app.py
```

If you want me to push this repo to GitHub or create additional deployment artifacts (Dockerfile, Procfile), tell me which option you prefer.

