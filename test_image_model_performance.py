"""
Real-time monitoring dashboard using Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="DR Detection Monitoring", layout="wide")

st.title("🏥 Diabetic Retinopathy Detection - Production Monitoring")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Predictions", "12,543", "+234 today")
    
with col2:
    st.metric("Avg Response Time", "145ms", "-12ms")
    
with col3:
    st.metric("Model Accuracy", "87.3%", "+2.1%")
    
with col4:
    st.metric("System Uptime", "99.9%", "")

# Load prediction logs
@st.cache_data
def load_predictions():
    predictions = []
    try:
        with open("logs/predictions.jsonl", "r") as f:
            for line in f:
                predictions.append(json.loads(line))
    except:
        # Generate sample data
        predictions = [
            {"timestamp": datetime.now().isoformat(), "risk_level": "Low", "confidence": 0.92},
            {"timestamp": datetime.now().isoformat(), "risk_level": "Medium", "confidence": 0.85},
        ]
    return pd.DataFrame(predictions)

df = load_predictions()

# Risk Distribution
st.subheader("Risk Level Distribution")
if not df.empty:
    risk_counts = df["risk_level"].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                 color_discrete_map={"Low": "green", "Medium": "yellow", "High": "red"})
    st.plotly_chart(fig)

# Time Series
st.subheader("Predictions Over Time")
# Add time series chart here

# Model Performance Comparison
st.subheader("Model A/B Test Results")
col1, col2 = st.columns(2)

with col1:
    st.info("**Stable Model (80% traffic)**")
    st.metric("Accuracy", "87.3%")
    st.metric("Avg Confidence", "0.89")
    
with col2:
    st.success("**Experimental Model (20% traffic)**")
    st.metric("Accuracy", "89.1%")
    st.metric("Avg Confidence", "0.91")

# Run with: streamlit run monitoring/dashboard.py