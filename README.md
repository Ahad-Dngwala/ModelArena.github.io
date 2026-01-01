# 🏟️ ModelArena: Enterprise ML Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**ModelArena** is a production-grade, low-code interface for training, comparing, and selecting Machine Learning models. Built for Data Scientists who need rapid experimentation with robust, enterprise-level reliability.

---

## 🚀 Key Features

### 🧪 **AutoML Training Laboratory**
- **Zero-Code Pipeline**: Upload your CSV, pick your target, and watch the magic happen.
- **Smart Preprocessing**: Automatic imputation, scaling, and One-Hot encoding via `sklearn.pipeline`.
- **Model Zoo**: Support for `Random Forest`, `XGBoost`, `Gradient Boosting`, `SVM`, `KNN`, and more.
- **Ensemble Power**: One-click **Voting Regressors/Classifiers** to combine top models.

### 📊 **Deep Insights Engine**
- **Leaderboard**: Instant comparison of all models sorted by your metric of choice (F1, Accuracy, R2, MSE).
- **Confusion Matrix**: Visual heatmap for classification error analysis.
- **Overfitting Detector**: Automatic alerts when training scores diverge significantly from validation scores.
- **Deep Dive Visuals**:
    - 🕸️ **Radar Charts**: Multi-metric comparison.
    - 📈 **ROC/AUC Curves**: Performance trade-off analysis.
    - 📉 **Residual Plots**: Homoscedasticity checks for regression.

### 🧠 **Explainability & Diagnostics**
- **SHAP Integration**: Understand *why* a model made a prediction with Kernel Explainer visualizations.
- **Feature Importance**: Native tree-based feature ranking.

### 🚢 **Production Ready**
- **Deployment Generator**: Auto-generates `FastAPI` code to serve your best model instantly.
- **Model Inspector**: dedicated page to audit `.pkl` files and visualize pipeline steps.

### 😊 **All Features**
- **Dataset Handling**: Upload CSV files, automatic detection of target and problem type.
- **Preprocessing**: Automated handling of missing values, categorical encoding, and scaling using Scikit-learn Pipelines.
- **Model Training**: Train multiple models efficiently.
- **Hyperparameter Tuning**: Interactive sliders for key model parameters.
- **Fair Comparison**: Consistent train/validation splits.
- **Insights**: Automated detection of overfitting/underfitting and best model highlights.
- **Visualizations**: Performance charts, feature importance, and more.
- **Model Inspector**: A dedicated viewer for `.pkl` files to inspect trained model structures and hyperparameters.
- **MLflow / Weights & Biases Integration**: Track every experiment run, hyperparameter set, and metric in a remote server.
- **Model Registry**: Version control for models (v1.0, v1.1) with "Staging" and "Production" tags.
- **SHAP & LIME**: Move beyond simple feature importance to explain *individual* predictions (e.g., "Why was Customer A denied a loan?").
- **Partial Dependence Plots (PDP)**: Visualize how a specific feature affects the outcome.
- **ydata-profiling**: Integrate automated comprehensive EDA reports before training.
- **Data Drift Detection**: Alert users if the new upload is significantly different from training data (using `evidently` or `alibi-detect`).
- **FastAPI Code Gen**: Auto-generate a `serving.py` file to accept JSON requests for the trained model.
- **Dockerization**: Auto-create a `Dockerfile` to containerize the app/model for cloud deployment.
- **AutoML**: Integrate frameworks like TPOT or H2O to search thousands of pipelines automatically.
- **Ensembling**: allow "Stacking" the top 3 models to create a super-learner.
- **Modular Design**: Separated concerns for data loading, preprocessing, and training.
- **Pipelines**: Used Scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage and ensure cleaner code.
- **State Management**: leveraged `st.session_state` to persist model results across page repaints.

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/model-arena.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

## 🏗️ Architecture

ModelArena is built on a modular stack designed for scalability:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Reactive web interface with custom CSS styling. |
| **ML Core** | `Scikit-Learn`, `XGBoost` | robust modeling and pipelining. |
| **Explainability** | `SHAP` | Game-theoretic feature attribution. |
| **Visualization** | `Plotly`, `Seaborn` | Interactive, exportable charts. |
| **Serialization** | `Joblib` | Efficient model saving/loading. |

---

## 📸 Workflow

1.  **Upload**: Drag & drop your dataset.
2.  **Config**: Select validation split and hyperparameters (or leave defaults).
3.  **Train**: Click **"Initiate Training Protocol"** and watch the progress bar.
4.  **Analyze**: Move to the **Insights** tab to pick the winner.
5.  **Deploy**: Download your `.pkl` model and the generated API code.

---

## 📝 Roadmap & Status

- [x] Core Training Pipeline (Classification/Regression)
- [x] Advanced Visualizations (Radar, ROC, Residuals)
- [x] SHAP Explainability
- [x] Voting Ensembles
- [x] Production Deployment Code Gen
- [ ] Automated Hyperparameter Optimization (Optuna) [Planned]

---

**Built with ❤️ for High-Performance ML Teams by Ahad Dangarvawala.**
