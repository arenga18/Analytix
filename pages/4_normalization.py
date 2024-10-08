import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sidebar import sidebar
import os

st.set_page_config(
    page_title="AnalytiX",
    page_icon="images/logo.png", layout="wide"
)

for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

st.markdown("""
    <style>
        .st-emotion-cache-mnu3yk{
            display: none !important;
        }
        .st-emotion-cache-q16mip{
            background:red !important;
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Membaca dataset
file_path = 'dataset.csv'

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

file_path = 'dataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    st.title("Normalization Data")
    st.write("*Normalization data is the process of scaling feature values in a dataset to ensure they have comparable ranges.*")
    st.write("-----")

    st.subheader("Normalization Options")
    scaler_option = st.radio("Select Scaler:", ("MinMaxScaler", "StandardScaler"))

    if st.button("Normalization"):
        if scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
            # Memilih kolom numerik untuk dinormalisasi
            numeric_columns = df.select_dtypes(include=np.number).columns
            scaled_data = scaler.fit_transform(df[numeric_columns])
            df[numeric_columns] = scaled_data
        elif scaler_option == "StandardScaler":
            scaler = StandardScaler()
            # Memilih kolom numerik untuk dinormalisasi kecuali kolom terakhir
            numeric_columns = df.select_dtypes(include=np.number).columns[:-1]
            scaled_data = scaler.fit_transform(df[numeric_columns])
            df[numeric_columns] = scaled_data

        df.to_csv(file_path, index=False)

        # Refresh page
        st.rerun()

    st.subheader("DataFrame after Normalization")
    st.write(df)
else:
    st.warning('Please upload the dataset first.')
