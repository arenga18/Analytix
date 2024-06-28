import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from sidebar import sidebar
import os

st.set_page_config(
    page_title="AnalyTIX",
    page_icon="images/logo.png", layout="wide"
)


for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

file_path = 'dataset.csv'
if os.path.exists(file_path):
    
    df = pd.read_csv(file_path)
    
    if not df.empty:
        st.title('Exploratory Data Analysis Automation')
        profile_report = df.profile_report(config_file="pages/config_py.yml", interactions=None)
        st_profile_report(profile_report)
    else:
        st.warning('The dataset is empty. Please upload a valid dataset first.')
else:
    st.warning('Please upload the dataset first.')
