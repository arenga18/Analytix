import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sidebar import sidebar

st.set_page_config(page_title = "AnalyTIX", page_icon = "📝", layout="wide")
for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)
        
# Membaca dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Judul aplikasi
st.title("Normalization Data")

# Pilihan Normalisasi di dalam body utama
st.subheader("Pilihan Normalisasi")
scaler_option = st.radio("Pilih Scaler:", ("MinMaxScaler", "StandardScaler"))

if st.button("Normalisasi"):
    if scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_option == "StandardScaler":
        scaler = StandardScaler()

    # Normalisasi data
    scaled_data = scaler.fit_transform(df.select_dtypes(include=np.number))
    df[df.select_dtypes(include=np.number).columns] = scaled_data
    
    # Menyimpan hasil normalisasi ke file CSV
    df.to_csv(file_path, index=False)
    
    # Refresh halaman untuk memperbarui DataFrame
    st.experimental_rerun()

# Menampilkan dataframe setelah normalisasi
st.subheader("DataFrame setelah Normalisasi")
st.write(df)
