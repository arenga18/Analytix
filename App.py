import streamlit as st
from streamlit_login_auth_ui.widgets import __login__

st.set_page_config(
    page_title="AnalyTIX",
    page_icon="📝",
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
    st.sidebar.page_link("App.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/1_Demo.py", label="Plotting")
    st.write("# Welcome to AnalyTIX! 👋")
    st.markdown(
        """
        **AnalyTIX** adalah sebuah aplikasi yang diracang untuk para Data Scientist yang ingin melakukan uji & analisa Machine Learning.
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [communityforums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car ImageDataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)"""
        )
