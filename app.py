import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="AutoML Pipeline", layout="wide")

st.title("🚀 Advanced ML Pipeline Dashboard")

# Sidebar
st.sidebar.header("Step 1: Configuration")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# File upload
st.header("📂 Data Input & PCA Visualization")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

# 🔥 FIXED CSV LOADING
def load_data(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding='latin-1')
        except:
            file.seek(0)
            return pd.read_csv(file, encoding='cp1252')

if uploaded_file:
    df = load_data(uploaded_file)

    st.write("### Preview of Dataset", df.head())

    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Select Target Feature", df.columns)
    with col2:
        default_feats = [c for c in df.columns if c != target_col]
        features = st.multiselect("Select Features", default_feats, default=default_feats)

    # PCA
    if features:
        X_pca = df[features].select_dtypes(include=[np.number]).fillna(0)
        if X_pca.shape[1] >= 2:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(X_pca)
            fig = px.scatter(x=comp[:,0], y=comp[:,1], color=df[target_col].astype(str))
            st.plotly_chart(fig, use_container_width=True)

    # Split
    st.header("✂ Data Split")
    test_size = st.slider("Test Size (%)", 10, 50, 20)

    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col]

    if problem_type == "Classification" and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    st.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Model
    st.header("🤖 Model Training")

    model_choice = st.selectbox("Model", ["Linear", "SVM", "Random Forest", "KMeans"])

    if model_choice == "Linear":
        model = LogisticRegression() if problem_type=="Classification" else LinearRegression()
    elif model_choice == "SVM":
        model = SVC() if problem_type=="Classification" else SVR()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier() if problem_type=="Classification" else RandomForestRegressor()
    else:
        model = KMeans(n_clusters=3)

    if st.button("Train Model"):
        cv = cross_validate(model, X_train, y_train, cv=5)
        st.write("Train Score:", np.mean(cv['train_score']) if 'train_score' in cv else "N/A")
        st.write("Validation Score:", np.mean(cv['test_score']))

        model.fit(X_train, y_train)
        st.success("Model Trained Successfully!")