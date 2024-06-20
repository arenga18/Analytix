import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sidebar import sidebar
import os

st.set_page_config(page_title="AnalyTIX", page_icon="📝", layout="wide")
for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

# Membaca dataset
file_path = 'dataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    st.title("Normalization Data")

    st.subheader("Normalization Options")
    scaler_option = st.radio("Select Scaler:", ("MinMaxScaler", "StandardScaler"))

    if st.button("Normalization"):
        if scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_option == "StandardScaler":
            scaler = StandardScaler()

        scaled_data = scaler.fit_transform(df.select_dtypes(include=np.number))
        df[df.select_dtypes(include=np.number).columns] = scaled_data

        df.to_csv(file_path, index=False)

        # refresh page
        st.experimental_rerun()
        
    st.subheader("DataFrame after Normalization")
    st.write(df)
else:
    st.warning('Please upload the dataset first.')
