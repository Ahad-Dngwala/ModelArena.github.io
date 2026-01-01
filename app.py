import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)

# Optional imports
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Page Setup
st.set_page_config(
    page_title="ModelArena Enterprise",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enterprise CSS Styling ---
st.markdown("""
    <style>
    /* Global Fonts & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Card Style for Metrics/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a202c;
        font-weight: 700;
    }
    h1 { margin-bottom: 1.5rem; }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 97, 255, 0.3);
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
        border-left: 4px solid #10b981;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state or isinstance(st.session_state.model_results, dict):
    st.session_state.model_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'models_store' not in st.session_state:
    st.session_state.models_store = {}

# --- Helper Functions ---

@st.cache_data
def load_data(file):
    if file is not None:
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

def detect_problem_type(df, target_col):
    if df[target_col].nunique() < 20 or df[target_col].dtype == 'object':
        return "Classification"
    return "Regression"

# --- Sidebar ---
with st.sidebar:
    st.title("🏢 ModelArena Pro")
    st.markdown("### Enterprise ML Dashboard")
    page = st.radio("Navigate", ["Model Training", "Insights & Selection", "Model Inspector"], label_visibility="collapsed")
    st.markdown("---")
    st.info("💡 **Tip:** Use the Insights page to download your production model.")

# --- PAGE 1: Model Training ---
if page == "Model Training":
    st.title("🧪 Model Training Laboratory")
    
    # 1. Dataset Handling
    with st.expander("📂 1. Data Selection & Profiling", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state.data = df
            
            if df is not None:
                st.markdown(f"**Data Status**: Loaded successfully. Shape: `{df.shape}`")
                st.dataframe(df.head())
                
                # Profiling
                if st.checkbox("Run Deep Data Profiling (ydata)"):
                    if PROFILING_AVAILABLE:
                        with st.spinner("Analyzing data distribution..."):
                            pr = ProfileReport(df, explorative=True, minimalist=True)
                            st_profile_report(pr)
                    else:
                        st.warning("Install `ydata-profiling` to enable this feature.")

                cols = df.columns.tolist()
                target_col = st.selectbox("Select Target Variable (Y)", cols, index=len(cols)-1)
                problem_type = detect_problem_type(df, target_col)
                
                # Type indicator pill
                color = "blue" if problem_type == "Classification" else "orange"
                st.markdown(f":{color}[**Detected Problem Type: {problem_type}**]")

    if st.session_state.data is not None:
        # 2. Config & Models
        with st.expander("⚙️ 2. Configuration & Hyperparameters", expanded=True):
            col_L, col_R = st.columns(2)
            
            with col_L:
                test_size = st.slider("Validation Split Ratio", 0.1, 0.4, 0.2, 0.05)
                # Feature selection could go here
            
            with col_R:
                st.markdown("### Model Zoo")
                base_models = ["Logistic/Linear Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Gradient Boosting"]
                if XGB_AVAILABLE:
                    base_models.append("XGBoost")
                
                model_options = base_models + ["Voting Ensemble (All Selected)"]
                
                models_to_train = st.multiselect("Select Models", model_options, default=["Logistic/Linear Regression", "Random Forest"])
            
            # Dynamic Hyperparams
            st.markdown("#### Hyperparameter Tuning")
            hyperparams = {}
            hp_col1, hp_col2, hp_col3 = st.columns(3)
            
            if any(m in models_to_train for m in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "Voting Ensemble (All Selected)"]):
                with hp_col1:
                    max_depth = st.slider("Max Depth (Trees)", 1, 30, 10)
            
            if any(m in models_to_train for m in ["Random Forest", "Gradient Boosting", "XGBoost", "Voting Ensemble (All Selected)"]):
                with hp_col2:
                    n_estimators = st.slider("N Estimators (Trees)", 50, 500, 100, step=50)
            
            if "KNN" in models_to_train or "Voting Ensemble (All Selected)" in models_to_train:
                with hp_col3:
                    n_neighbors = st.slider("K Neighbors", 3, 21, 5, step=2)

        # 3. Action
        if st.button("🚀 Initiating Training Protocol", type="primary"):
            # Progress Elements
            progress_bar = st.progress(0, text="Initializing core systems...")
            status_text = st.empty()
            
            try:
                df = st.session_state.data
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Preprocessing
                num_feats = X.select_dtypes(include=['int64', 'float64']).columns
                cat_feats = X.select_dtypes(include=['object', 'category']).columns
                
                num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
                cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
                
                preprocessor = ColumnTransformer([('num', num_pipe, num_feats), ('cat', cat_pipe, cat_feats)])
                
                if problem_type == "Classification" and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.session_state.update({'X_train': X_train, 'y_test': y_test, 'X_test': X_test, 'problem_type': problem_type, 'preprocessor': preprocessor})

                from sklearn.base import clone
                
                # Model Factory
                def get_model(name):
                    try:
                        if problem_type == "Classification":
                            if name == "Logistic/Linear Regression": return LogisticRegression(max_iter=2000)
                            if name == "Decision Tree": return DecisionTreeClassifier(max_depth=max_depth)
                            if name == "Random Forest": return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                            if name == "SVM": return SVC(probability=True)
                            if name == "KNN": return KNeighborsClassifier(n_neighbors=n_neighbors)
                            if name == "Gradient Boosting": return GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
                            if name == "XGBoost" and XGB_AVAILABLE: return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, eval_metric='logloss')
                        else:
                            if name == "Logistic/Linear Regression": return LinearRegression()
                            if name == "Decision Tree": return DecisionTreeRegressor(max_depth=max_depth)
                            if name == "Random Forest": return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                            if name == "SVM": return SVR()
                            if name == "KNN": return KNeighborsRegressor(n_neighbors=n_neighbors)
                            if name == "Gradient Boosting": return GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
                            if name == "XGBoost" and XGB_AVAILABLE: return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    except Exception as e:
                        st.warning(f"Error creating model {name}: {e}")
                    return None

                # Prepare Ensemble
                status_text.markdown("🧩 **Structuring Ensemble Architecture...**")
                estimators = []
                try:
                    single_models = [m for m in models_to_train if "Voting" not in m]
                    for m in single_models:
                        mod = get_model(m)
                        if mod: estimators.append((m, mod))
                except Exception as e:
                    st.warning(f"⚠️ Error preparing ensemble elements: {e}")
                
                results = []
                models_store = {}
                
                total_steps = len(models_to_train)
                
                for i, name in enumerate(models_to_train):
                    # Update Progress
                    progress = (i / total_steps)
                    progress_bar.progress(progress, text=f"Training {name} ({int(progress*100)}%)")
                    status_text.markdown(f"⚙️ **Training Active:** `{name}` running optimization...")
                    
                    try:
                        if name == "Voting Ensemble (All Selected)":
                            if not estimators: continue
                            # Clone estimators to ensure freshness
                            from sklearn.base import clone
                            voting_estimators = [(n, get_model(n)) for n, _ in estimators] 
                            model = VotingClassifier(voting_estimators, voting='soft') if problem_type == "Classification" else VotingRegressor(voting_estimators)
                        else:
                            model = get_model(name)
                        
                        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
                        pipeline.fit(X_train, y_train)
                        
                        y_pred = pipeline.predict(X_test)
                        y_train_pred = pipeline.predict(X_train)
                        
                        res = {'Model': name}
                        if problem_type == "Classification":
                            res.update({
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'Train Accuracy': accuracy_score(y_train, y_train_pred)
                            })
                        else:
                            res.update({
                                'R2 Score': r2_score(y_test, y_pred),
                                'MSE': mean_squared_error(y_test, y_pred),
                                'Train R2': r2_score(y_train, y_train_pred)
                            })
                        results.append(res)
                        models_store[name] = pipeline
                    except Exception as model_error:
                        st.warning(f"⚠️ Could not train **{name}**: {model_error}")
                
                # Finalize
                progress_bar.progress(1.0, text="Finalizing leaderboard...")
                status_text.success("Training Protocol Complete!")
                
                st.session_state.model_results = pd.DataFrame(results)
                st.session_state.models_store = models_store
                st.toast("Analysis Complete!", icon="✅")
                
            except Exception as e:
                st.error(f"Training Failed: {e}")

    # Results Preview
    if st.session_state.model_results is not None and not st.session_state.model_results.empty:
        st.markdown("### 🏆 Leaderboard Snapshot")
        st.dataframe(st.session_state.model_results.style.highlight_max(axis=0))

# --- PAGE 2: Insights ---
elif page == "Insights & Selection":
    st.title("📊 Model Analytics & Selection")
    
    if st.session_state.model_results is None or st.session_state.model_results.empty:
        st.info("No models trained. Go to 'Model Training' tab.")
    else:
        results = st.session_state.model_results
        ptype = st.session_state.get('problem_type', "Classification")
        
        # 1. Best Model Logic
        st.sidebar.markdown("### 🎯 Optimization Goal")
        if ptype == "Classification":
            metric_choice = st.sidebar.selectbox("Optimize For:", ["F1 Score", "Accuracy", "Recall", "Precision"])
        else:
            metric_choice = st.sidebar.selectbox("Optimize For:", ["R2 Score", "MSE"])
        
        # Sorting
        ascending = True if metric_choice == "MSE" else False
        best_row = results.sort_values(metric_choice, ascending=ascending).iloc[0]
        best_name = best_row['Model']
        
        # Display Best Model
        st.markdown(f"""
        <div class="css-1r6slb0">
            <h3>👑 Champion Model: <span style="color:#0061ff">{best_name}</span></h3>
            <p>Selected based on highest <b>{metric_choice}</b> of <b>{best_row[metric_choice]:.4f}</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        if ptype == "Classification":
            col1.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
            col2.metric("F1 Score", f"{best_row['F1 Score']:.3f}")
            col3.metric("Precision", f"{best_row['Precision']:.3f}")
            col4.metric("Recall", f"{best_row['Recall']:.3f}")
        else:
            col1.metric("R2 Score", f"{best_row['R2 Score']:.3f}")
            col2.metric("MSE", f"{best_row['MSE']:.3f}")
        
        # 2. Visuals
        st.subheader("📈 Performance Benchmarking")
        tab1, tab2, tab3, tab4 = st.tabs(["Metric Comparison", "Bias-Variance Analysis", "Confusion Matrix", "🔍 Deep Dive Charts"])
        
        with tab1:
            metrics_to_plot = ["Accuracy", "F1 Score", "Recall"] if ptype=="Classification" else ["R2 Score", "Train R2"]
            df_melt = results.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
            fig = px.bar(df_melt, x="Model", y="Score", color="Metric", barmode='group', template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            train_col = 'Train Accuracy' if ptype=="Classification" else 'Train R2'
            test_col = 'Accuracy' if ptype=="Classification" else 'R2 Score'
            
            fig2 = go.Figure(data=[
                go.Bar(name='Train', x=results['Model'], y=results[train_col], marker_color='#cbd5e0'),
                go.Bar(name='Validation', x=results['Model'], y=results[test_col], marker_color='#0061ff')
            ])
            fig2.update_layout(title="Overfitting Detection (Train vs Val)", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
            
            for index, row in results.iterrows():
                diff = row[train_col] - row[test_col]
                if diff > 0.15: st.warning(f"⚠️ {row['Model']} is Overfitting (Gap: {diff:.2f})")
                
        with tab3:
            if ptype == "Classification":
                st.write(f"Confusion Matrix for **{best_name}**")
                model_pipe = st.session_state.models_store[best_name]
                y_pred = model_pipe.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                   labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion Matrix is for Classification only.")

        with tab4:
            st.markdown("### 🔬 Advanced Model Diagnostics")
            
            # Radar Chart
            st.markdown("#### 🕸️ Model Radar Comparison (spider plot)")
            if ptype == "Classification":
                radar_metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Train Accuracy']
            else:
                radar_metrics = ['R2 Score', 'Train R2'] # MSE is different scale, maybe exclude or normalize
            
            fig_radar = go.Figure()
            
            for index, row in results.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[m] for m in radar_metrics],
                    theta=radar_metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Multi-Metric Comparison"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # ROC / Residuals
            st.markdown("#### 🎯 Prediction Analysis")
            best_pipe = st.session_state.models_store[best_name]
            
            if ptype == "Classification":
                if hasattr(best_pipe, "predict_proba"):
                    from sklearn.metrics import roc_curve, auc
                    y_prob = best_pipe.predict_proba(st.session_state.X_test)
                    
                    # Handle Binary vs Multi-class (Simplified for Binary here, or macro avg)
                    if y_prob.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc = px.area(
                            x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})",
                            labels=dict(x="False Positive Rate", y="True Positive Rate"),
                            color_discrete_sequence=["#0061ff"]
                        )
                        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.info("ROC Curve available for binary classification.")
                else:
                    st.warning("Selected model does not support probability output for ROC.")
            
            else: # Regression
                y_pred = best_pipe.predict(st.session_state.X_test)
                residuals = st.session_state.y_test - y_pred
                
                fig_res = px.scatter(
                    x=y_pred, y=residuals,
                    labels={'x': 'Predicted Values', 'y': 'Residuals (Actual - Predicted)'},
                    title="Residual Plot (Homoscedasticity Check)"
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)
                
                fig_pred = px.scatter(
                    x=st.session_state.y_test, y=y_pred,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title="Actual vs Predicted"
                )
                fig_pred.add_shape(type='line', line=dict(dash='dash', color='gray'), 
                                   x0=st.session_state.y_test.min(), y0=st.session_state.y_test.min(),
                                   x1=st.session_state.y_test.max(), y1=st.session_state.y_test.max())
                st.plotly_chart(fig_pred, use_container_width=True)

            # Feature Importance
            if hasattr(best_pipe.named_steps['model'], 'feature_importances_'):
                st.markdown("#### 🌲 Feature Importance")
                importances = best_pipe.named_steps['model'].feature_importances_
                feature_names = st.session_state.X_train.columns # Approximation (pre-transform names)
                # Note: Correct names after OneHot is hard, simplistic approach for now:
                if len(importances) == len(feature_names):
                     feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                     fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (Raw Features)")
                     st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("Feature importance available (dimension mismatch with raw features, likely due to OneHotEncoding). Check SHAP for detailed view.")

        # 3. Explainability
        st.subheader("🔍 Explainability (SHAP)")
        if st.checkbox("Calculate SHAP Values"):
            best_pipe = st.session_state.models_store[best_name]
            try:
                # Extract components
                model_core = best_pipe.named_steps['model']
                preprocessor = best_pipe.named_steps['preprocessor']
                X_sample = preprocessor.transform(st.session_state.X_train)[:100] # Limit to 100 for speed
                
                with st.spinner("Computing SHAP values..."):
                    explainer = shap.KernelExplainer(model_core.predict, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    
                    fig_shap, ax = plt.subplots()
                    if isinstance(shap_values, list): # Multiclass
                        shap.summary_plot(shap_values[0], X_sample, show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP not available for this model type specifically: {e}")

        # 4. Deployment
        st.markdown("### 🚢 Production Export")
        with st.expander("Get Deployment Code"):
            st.code(f"""
# FastAPI Deployment for {best_name}
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    return {{'prediction': model.predict(df).tolist()}}
            """, language='python')
            
            buffer = io.BytesIO()
            joblib.dump(st.session_state.models_store[best_name], buffer)
            buffer.seek(0)
            st.download_button(f"Download {best_name} (.pkl)", data=buffer, file_name="model.pkl")

# --- PAGE 3: Inspector ---
elif page == "Model Inspector":
    st.title("🔎 Model Inspector")
    uploaded_pkl = st.file_uploader("Upload .pkl", type="pkl")
    
    if uploaded_pkl:
        model = joblib.load(uploaded_pkl)
        st.success(f"Loaded: {type(model).__name__}")
        
        if isinstance(model, Pipeline):
            st.subheader("Pipeline Steps")
            # FIX: Convert objects to string to avoid Arrow serialization error
            steps = [(str(k), str(v)) for k, v in model.steps]
            st.table(pd.DataFrame(steps, columns=["Step", "Component"]))
            
        if hasattr(model, "get_params"):
            with st.expander("View Hyperparameters"):
                st.json({k: str(v) for k, v in model.get_params().items()})
