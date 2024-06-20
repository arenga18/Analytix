import streamlit as st
import pandas as pd
import numpy as np
from sidebar import sidebar

st.set_page_config(page_title = "AnalyTIX", page_icon = "📝", layout="wide")
for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

# Membaca dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Judul aplikasi
st.title("Data Cleaning")

# Membagi halaman menjadi dua kolom
col1, col2 = st.columns((1, 1), gap="large")

with col1:
    st.subheader("Data Duplikat")
    # Menampilkan data duplikat
    duplicates = df[df.duplicated()]
    num_duplicates = duplicates.shape[0]
    percent_duplicates = (num_duplicates / df.shape[0]) * 100
    
    st.write(f"Jumlah data duplikat: {num_duplicates}")
    st.write(f"Persentase data duplikat: {percent_duplicates:.2f}%")
    st.write(duplicates)
    
    if not duplicates.empty:
        if st.button("Hapus Data Duplikat"):
            df = df.drop_duplicates()
            df.to_csv(file_path, index=False)  # Menyimpan kembali ke file CSV
            st.write("Data duplikat telah dihapus.")
            st.experimental_rerun()
    else:
        st.write("Tidak ada data duplikat yang ditemukan.")

with col2:
    st.subheader("Data Null")
    # Menampilkan data yang mengandung nilai null
    null_data = df[df.isnull().any(axis=1)]
    num_null = null_data.shape[0]
    percent_null = (num_null / df.shape[0]) * 100
    
    st.write(f"Jumlah data null: {num_null}")
    st.write(f"Persentase data null: {percent_null:.2f}%")
    st.write(null_data)
    
    if not null_data.empty:
        if st.button("Hapus Data Null"):
            df = df.dropna()
            df.to_csv(file_path, index=False)  # Menyimpan kembali ke file CSV
            st.write("Data yang mengandung nilai null telah dihapus.")
            st.experimental_rerun()
    else:
        st.write("Tidak ada data null yang ditemukan.")

# Menampilkan dataframe
st.write("\n")
st.subheader("DataFrame")
st.write(df)