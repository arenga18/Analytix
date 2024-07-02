import streamlit as st
import pandas as pd
import numpy as np
import os
from sidebar import sidebar
from sidebar import remove_github_source

st.set_page_config(
    page_title="AnalytiX",
    page_icon="images/logo.png", layout="wide"
)
remove_github_source

for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

# Membaca dataset
file_path = 'dataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Judul aplikasi
    st.title("Data Cleaning")
    st.write("*Data cleaning is to prepare and correct data by removing or correcting errors, inconsistencies, and inaccuracies. This process involves removing duplicate data and handling missing data to ensure the dataset is accurate, complete, and ready for analysis.*")
    st.write("-----")

    # Membagi halaman menjadi dua kolom
    col1, col2 = st.columns((1, 1), gap="large")

    with col1:
        st.subheader("Duplicate data")
        # Menampilkan data duplikat
        duplicates = df[df.duplicated()]
        num_duplicates = duplicates.shape[0]
        percent_duplicates = (num_duplicates / df.shape[0]) * 100

        st.write(f"Number of duplicate data: {num_duplicates}")
        st.write(f"Percentage of duplicate data: {percent_duplicates:.2f}%")
        st.write(duplicates)

        if not duplicates.empty:
            if st.button("Delete Duplicate Data"):
                df = df.drop_duplicates()
                df.to_csv(file_path, index=False)  # Menyimpan kembali ke file CSV
                st.write("Duplicate data was removed.")
                st.experimental_rerun()
        else:
            st.write("No duplicate data found.")

    with col2:
        st.subheader("Missing Data")

        # Check Null Data
        null_data = df[df.isnull().any(axis=1)]
        num_null = null_data.shape[0]
        percent_null = (num_null / df.shape[0]) * 100

        # Check Missing Data
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()
        percent_missing = (total_missing / df.size) * 100

        st.write(f"Number of missing data: {total_missing}")
        st.write(f"Percentage of missing data: {percent_missing:.2f}%")
        st.write(null_data)

        if not null_data.empty:
            if st.button("Delete Missing Data"):
                df = df.dropna()
                df.to_csv(file_path, index=False)  # Menyimpan kembali ke file CSV
                st.write("Missing data was removed.")
                st.experimental_rerun()
        else:
            st.write("No Missing data found.")

    # Menampilkan dataframe
    st.write("\n")
    st.subheader("DataFrame")
    st.write(df)
else:
    st.warning('Please upload the dataset first.')
