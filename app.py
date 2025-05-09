import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Health Data Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("Health Data Prediction Dashboard")
st.markdown("""
This dashboard analyzes heart disease prediction data and displays various visualizations and model comparisons.
""")

# Load data
@st.cache_data
def load_data():
    try:
        # Load heart data with explicit column names
        heart_data = pd.read_csv('heart.csv')
        
        # Load O2 saturation data as a single column
        o2_data = pd.read_csv('o2Saturation.csv', header=None, names=['o2_saturation'])
        
        return heart_data, o2_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

try:
    heart_data, o2_data = load_data()
    
    if heart_data is None or o2_data is None:
        st.error("Failed to load data. Please check if the data files exist and are properly formatted.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Model Analysis", "Visualizations"])
    
    if page == "Data Overview":
        st.header("Data Overview")
        
        # Display basic statistics
        st.subheader("Heart Data Statistics")
        st.write(heart_data.describe())
        
        # Display first few rows
        st.subheader("Heart Data Sample")
        st.dataframe(heart_data.head())
        
        # Data shape
        st.subheader("Dataset Information")
        st.write(f"Number of rows: {heart_data.shape[0]}")
        st.write(f"Number of columns: {heart_data.shape[1]}")
        
        # O2 Saturation Statistics
        st.subheader("O2 Saturation Statistics")
        st.write(o2_data.describe())
        
    elif page == "Model Analysis":
        st.header("Model Analysis")
        
        # Prepare data for modeling
        X = heart_data.drop('output', axis=1)
        y = heart_data['output']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = rmse
        
        # Display model comparison
        st.subheader("Model Comparison (RMSE)")
        fig = px.bar(
            x=list(results.keys()),
            y=list(results.values()),
            labels={'x': 'Model', 'y': 'RMSE'},
            title='Model Performance Comparison'
        )
        st.plotly_chart(fig)
        
        # Feature importance for Random Forest
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.subheader("Feature Importance (Random Forest)")
        fig = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title='Feature Importance from Random Forest Model'
        )
        st.plotly_chart(fig)
        
    else:  # Visualizations
        st.header("Data Visualizations")
        
        # Age Distribution
        st.subheader("Age Distribution")
        fig = px.histogram(
            heart_data,
            x='age',
            nbins=20,
            title='Age Distribution',
            labels={'age': 'Age', 'count': 'Count'}
        )
        st.plotly_chart(fig)
        
        # Gender Distribution
        st.subheader("Gender Distribution")
        gender_counts = heart_data['sex'].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=['Female', 'Male'],
            title='Gender Distribution'
        )
        st.plotly_chart(fig)
        
        # Cholesterol Distribution
        st.subheader("Cholesterol Levels")
        fig = px.histogram(
            heart_data,
            x='chol',
            nbins=20,
            title='Cholesterol Distribution',
            labels={'chol': 'Cholesterol', 'count': 'Count'}
        )
        st.plotly_chart(fig)
        
        # Target Variable Distribution
        st.subheader("Target Variable Distribution")
        target_counts = heart_data['output'].value_counts()
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            title='Target Variable Distribution',
            labels={'x': 'Output', 'y': 'Count'}
        )
        st.plotly_chart(fig)
        
        # O2 Saturation Distribution
        st.subheader("O2 Saturation Distribution")
        fig = px.histogram(
            o2_data,
            x='o2_saturation',
            nbins=20,
            title='O2 Saturation Distribution',
            labels={'o2_saturation': 'O2 Saturation', 'count': 'Count'}
        )
        st.plotly_chart(fig)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please make sure the data files (heart.csv and o2Saturation.csv) are in the correct directory.") 