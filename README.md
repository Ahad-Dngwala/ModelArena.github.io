# ModelArena: Streamlit-Based Model Comparison & Selection Dashboard

## Overview
ModelArena is a production-quality Streamlit web application designed for fair comparison, training, and selection of machine learning models. It provides an interactive interface to upload datasets, preprocess data, train multiple models (Logistic/Linear Regression, Decision Trees, Random Forests), and analyze their performance to select the best one.

## Features
- **Dataset Handling**: Upload CSV files, automatic detection of target and problem type.
- **Preprocessing**: Automated handling of missing values, categorical encoding, and scaling using Scikit-learn Pipelines.
- **Model Training**: Train multiple models efficiently.
- **Hyperparameter Tuning**: Interactive sliders for key model parameters.
- **Fair Comparison**: Consistent train/validation splits.
- **Insights**: Automated detection of overfitting/underfitting and best model highlights.
- **Visualizations**: Performance charts, feature importance, and more.
- **Model Inspector**: A dedicated viewer for `.pkl` files to inspect trained model structures and hyperparameters.

## Setup & Execution

### 1. Environment Setup
It is recommended to use a virtual environment.

```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```
Acccess the app at `http://localhost:8501`.

## Enterprise Roadmap (Future Enhancements)
To take this application to an enterprise level, the following features are recommended:

### 1. MLOps & Tracking
- **MLflow / Weights & Biases Integration**: Track every experiment run, hyperparameter set, and metric in a remote server.
- **Model Registry**: Version control for models (v1.0, v1.1) with "Staging" and "Production" tags.

### 2. Advanced Explainability (XAI)
- **SHAP & LIME**: Move beyond simple feature importance to explain *individual* predictions (e.g., "Why was Customer A denied a loan?").
- **Partial Dependence Plots (PDP)**: Visualize how a specific feature affects the outcome.

### 3. Data Robustness
- **ydata-profiling**: Integrate automated comprehensive EDA reports before training.
- **Data Drift Detection**: Alert users if the new upload is significantly different from training data (using `evidently` or `alibi-detect`).

### 4. Deployment Automation
- **FastAPI Code Gen**: Auto-generate a `serving.py` file to accept JSON requests for the trained model.
- **Dockerization**: Auto-create a `Dockerfile` to containerize the app/model for cloud deployment.

### 5. Advanced Modeling
- **AutoML**: Integrate frameworks like TPOT or H2O to search thousands of pipelines automatically.
- **Ensembling**: allow "Stacking" the top 3 models to create a super-learner.

## Key Learnings & Engineering
- **Modular Design**: Separated concerns for data loading, preprocessing, and training.
- **Pipelines**: Used Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage and ensure cleaner code.
- **State Management**: leveraged `st.session_state` to persist model results across page repaints.
