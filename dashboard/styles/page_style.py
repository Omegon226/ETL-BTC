import streamlit as st


def set_page_config_wide():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📊",
        layout="wide"
    )


def set_page_config_centered():
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📊",
        layout="centered"
    )


def disable_header_and_footer():
    hide_streamlit_style = """
            <style>
                /* Hide the Streamlit header and menu */
                header {visibility: hidden;}
                /* Optionally, hide the footer */
                .streamlit-footer {display: none;}
                /* Hide your specific div class, replace class name with the one you identified */
                .st-emotion-cache-uf99v8 {display: none;}
            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)