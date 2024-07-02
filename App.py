import streamlit as st
import os
import pandas as pd
from streamlit_login_auth_ui.widgets import __login__
from sidebar import sidebar

st.set_page_config(
    page_title="AnalytiX",
    page_icon="images/logo.png"
)

st.markdown("""
    <style>
        .st-emotion-cache-1y4p8pa {
            max-width: 55rem !important;
        }
        .st-emotion-cache-d0k6px{
            display:none;
        }
        .st-emotion-cache-13ln4jf{
            max-width: 55rem !important;
        }
        .stActionButton .st-emotion-cache-mnu3yk ef3psqc5{
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

        
__login__obj = __login__(
    auth_token="courier_auth_token",
    company_name="Shims",
    width=200,
    height=250,
    logout_button_name="Logout",
    hide_menu_bool=False,
    hide_footer_bool=False,
    lottie_url="https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json",
)

LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:

    st.markdown("""
    <style>
        .st-emotion-cache-1y4p8pa {
            max-width: 50rem !important;
        }
        .st-emotion-cache-d0k6px{
            display:none;
        }
        .st-emotion-cache-k7vsyb h1{
            padding-top: 0 !important;
        }
        .st-emotion-cache-13ln4jf{
            max-width: 50rem !important;
        }
        .st-emotion-cache-1jzia57 h1{
            padding-top: 0 !important;
        }
        .st-emotion-cache-1vxmjmh, .st-emotion-cache-1vxmjmh{
            display:none;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

    for page_link, label, icon in zip(sidebar['page_link'], sidebar['label'], sidebar['icon']):
        st.sidebar.page_link(page_link, label=label, icon=icon)

    st.title("WELCOME TO ANALYTIX! 👋")
    st.write(
        """
        **AnalytiX** is an application designed for data scientists aiming to perform thorough testing and analysis in machine learning. It offers a user-friendly interface with a wide array of tools and features.
        """
    )
    st.write("Let's get started!")
    
    file = st.file_uploader('Upload CSV Dataset Here')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
    
    if os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)
        st.dataframe(df)

        if st.button("Delete Dataset"):
            os.remove('dataset.csv')
            st.info("Dataset has been deleted.")
            st.experimental_rerun()
    else:
        df = pd.DataFrame()
   
