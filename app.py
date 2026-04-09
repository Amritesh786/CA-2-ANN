import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

# --- Page Config & Esthetics ---
st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .step-header { color: #1f77b4; font-weight: bold; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Advanced ML Pipeline Dashboard")

# --- Step 1: Problem Type ---
st.sidebar.header("Step 1: Configuration")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# --- Step 2: Data Input ---
st.header("📂 Data Input & PCA Visualization")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Feature", df.columns)
    with col2:
        features = st.multiselect("Select Features for Analysis", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

    if features:
        # PCA Visualization
        X_pca = df[features].select_dtypes(include=[np.number]).fillna(0)
        if X_pca.shape[1] >= 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_pca)
            fig_pca = px.scatter(components, x=0, y=1, color=df[target_col].astype(str),
                                 title="Data Shape (PCA Projection)", labels={'0': 'PC1', '1': 'PC2'})
            st.plotly_chart(fig_pca, use_container_width=True)

    # --- Step 3: EDA ---
    st.divider()
    st.header("📊 Exploratory Data Analysis (EDA)")
    if st.checkbox("Run EDA"):
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.write("Statistics", df.describe())
        with col_eda2:
            st.write("Missing Values", df.isnull().sum())
        
        fig_corr = px.imshow(df.select_dtypes(include=[np.number]).corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig_corr)

    # --- Step 4: Data Engineering & Cleaning ---
    st.divider()
    st.header("🛠 Data Engineering")
    
    # Imputation
    impute_method = st.selectbox("Imputation Method", ["None", "Mean", "Median", "Mode"])
    if impute_method != "None":
        for col in df.select_dtypes(include=[np.number]).columns:
            if impute_method == "Mean": df[col].fillna(df[col].mean(), inplace=True)
            elif impute_method == "Median": df[col].fillna(df[col].median(), inplace=True)
            elif impute_method == "Mode": df[col].fillna(df[col].mode()[0], inplace=True)

    # Outlier Detection
    outlier_method = st.selectbox("Outlier Detection Method", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
    outlier_indices = []

    if outlier_method != "None":
        numeric_df = df[features].select_dtypes(include=[np.number])
        if outlier_method == "IQR":
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            outlier_indices = numeric_df[((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)].index
        elif outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.1)
            preds = iso.fit_predict(numeric_df.fillna(0))
            outlier_indices = df.index[preds == -1]

        st.warning(f"Detected {len(outlier_indices)} outliers.")
        if len(outlier_indices) > 0:
            if st.button("Remove Outliers"):
                df = df.drop(outlier_indices).reset_index(drop=True)
                st.success("Outliers Removed!")

    # --- Step 5: Feature Selection ---
    st.divider()
    st.header("🎯 Feature Selection")
    fs_method = st.multiselect("Select Selection Methods", ["Variance Threshold", "Correlation", "Information Gain"])
    
    selected_fs_features = features.copy()
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col]
    if pd.api.types.is_string_dtype(y):
        y = LabelEncoder().fit_transform(y)

    if "Variance Threshold" in fs_method:
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(X)
        selected_fs_features = list(X.columns[selector.get_support()])
        st.write(f"After Variance Threshold: {selected_fs_features}")

    # --- Step 6: Data Split ---
    st.divider()
    st.header("✂ Data Split")
    test_size = st.slider("Test Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(df[selected_fs_features], y, test_size=test_size/100, random_state=42)
    st.info(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # --- Step 7: Model Selection & Training ---
    st.divider()
    st.header("🤖 Model Selection & K-Fold")
    
    model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "Random Forest", "K-Means"])
    k_val = st.number_input("Value of K for Cross-Validation", min_value=2, max_value=10, value=5)

    # Model Object Mapping
    model = None
    if model_choice == "Linear/Logistic Regression":
        model = LogisticRegression() if problem_type == "Classification" else LinearRegression()
    elif model_choice == "SVM":
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
    elif model_choice == "K-Means":
        model = KMeans(n_clusters=st.slider("Clusters", 2, 10, 3))

    if st.button("Train & Validate"):
        # K-Fold
        cv_results = cross_validate(model, X_train, y_train, cv=k_val, return_train_score=True)
        st.write(f"**Mean CV Train Score:** {cv_results['train_score'].mean():.4f}")
        st.write(f"**Mean CV Test Score:** {cv_results['test_score'].mean():.4f}")
        
        # Checking for Overfitting
        train_test_diff = cv_results['train_score'].mean() - cv_results['test_score'].mean()
        if train_test_diff > 0.15:
            st.error("Potential Overfitting Detected!")
        elif cv_results['train_score'].mean() < 0.6:
            st.warning("Potential Underfitting Detected.")
        else:
            st.success("Model seems balanced.")

        # Performance Metrics
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if problem_type == "Classification":
            st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        else:
            st.metric("Test R2 Score", f"{r2_score(y_test, y_pred):.2f}")

    # --- Step 9: Hyperparameter Tuning ---
    st.divider()
    st.header("⚙ Hyperparameter Tuning")
    if st.checkbox("Enable Tuning"):
        tune_type = st.radio("Search Method", ["Grid Search", "Random Search"])
        # Example for Random Forest
        if model_choice == "Random Forest":
            params = {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]}
            search = GridSearchCV(model, params, cv=3) if tune_type == "Grid Search" else RandomizedSearchCV(model, params, cv=3)
            search.fit(X_train, y_train)
            st.write("Best Parameters:", search.best_params_)
            st.write("Best Score:", search.best_score_)