import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sidebar import sidebar

st.set_page_config(
    page_title="AnalyTIX",
    page_icon="images/logo.png", layout="wide"
)

for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

# Fungsi untuk melakukan label encoding pada seluruh dataset
def encode_dataset(df):
    st.title("Encode Dataset")
    st.write("*Encoding data in machine learning is the process of transforming raw data, such as text , string, or categories, into a numerical format that can be understood and processed by machine learning algorithms.*")
    st.write("-----")

    # Memeriksa kolom yang memiliki nilai string
    string_columns = df.select_dtypes(include=['object']).columns

    if len(string_columns) > 0:
        st.subheader("Columns with String Values:")
        # Membuat dataframe untuk menampilkan isi kolom yang memiliki nilai string
        string_columns_df = df[string_columns]
        st.dataframe(string_columns_df)

        if st.button("Encode Data"):
            # Inisialisasi LabelEncoder
            encoder = LabelEncoder()
            # Loop melalui semua kolom dengan tipe data object (string)
            for column in string_columns:
                df[column] = encoder.fit_transform(df[column])

            # Simpan dataset yang telah diencode ke file CSV
            df.to_csv(file_path, index=False)

            # Refresh page
            st.experimental_rerun()
    else:
        st.warning("No columns with string values found. No encoding necessary.")
        st.write(df)

# Membaca dataset
file_path = 'dataset.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    encode_dataset(df)
else:
    st.warning('Please upload the dataset first.')
