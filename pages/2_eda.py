import pickle
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(page_title = "AnalyTIX", page_icon = "📝")
st.sidebar.page_link('App.py', label= 'Home', icon = None)
st.sidebar.page_link('pages/1_Demo.py', label= 'Demo', icon = None)
st.sidebar.page_link('pages/2_eda.py', label= 'EDA', icon = None)

# Muat DataFrame 'df' dari file pickle
with open('df.pickle', 'rb') as f:
    df = pickle.load(f)
    
if not df.empty:
    st.title('Exploratory Data Analysis Automation')
    profile_report = df.profile_report()
    st_profile_report(profile_report)
else:
    st.warning('Please upload the dataset first')