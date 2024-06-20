import streamlit as st
import pandas as pd
import numpy as np
from sidebar import sidebar
import plotly.express as px
import os

st.set_page_config(page_title="AnalyTIX", page_icon="📝", layout="wide")
for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

# Cek keberadaan file dataset.csv
file_path = 'dataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # App title
    st.title("Scatter Plot Visualization")

    col1, col2 = st.columns((1, 3), gap="large")

    with col1:    
        st.write(f"")    
        st.write(f"")    
        # Scatter plot selection
        x_axis = st.selectbox("Select X-axis", options=df.columns)
        y_axis = st.selectbox("Select Y-axis", options=df.columns)

    with col2: 
        # Plot using Plotly Express for better interactivity
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter plot of {x_axis} and {y_axis}")
        fig.update_traces(marker=dict(color='#ff6347'))

        st.plotly_chart(fig)
else:
    st.warning('Please upload the dataset first.')
