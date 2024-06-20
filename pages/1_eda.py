import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from sidebar import sidebar

st.set_page_config(page_title = "AnalyTIX", page_icon = "📝", layout="wide")
for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
    st.sidebar.page_link(page_link, label=label, icon=icon)

# Muat DataFrame 'df' dari file pickle
df = pd.read_csv('dataset.csv')
    
if not df.empty:
    st.title('Exploratory Data Analysis Automation')
    profile_report = df.profile_report(config_file="pages/config_py.yml", interactions=None)
    st_profile_report(profile_report)
else:
    st.warning('Please upload the dataset first')   