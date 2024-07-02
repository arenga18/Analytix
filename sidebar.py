import streamlit as st

sidebar = {
    'page_link': [
        'App.py',
        'pages/1_eda.py',
        'pages/2_data_cleaning.py',
        'pages/3_encode_data.py',
        'pages/4_normalization.py',
        'pages/5_plot.py',
        'pages/6_ml.py'
    ],
    'label': [
        'Home',
        'EDA Automation',
        'Data Cleaning',
        'Data Encoding',
        'Data Normalization',
        'Scatter Plot',
        'ML'
    ],
    'icon': [
        '🏡',
        '📊',
        '🧹',
        '🔢',
        '⚖️',
        '🪢',
        '🖥️'
    ]
        
}


remove_github_source = st.markdown("""
            <style>
                .st-emotion-cache-mnu3yk ef3psqc5, 
                .st-emotion-cache-mnu3yk ef3psqc5 {
                    display: none !important;
                }
            </style>
            """,
            unsafe_allow_html=True)



