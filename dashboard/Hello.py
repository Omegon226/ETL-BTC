import streamlit as st
from dotenv import load_dotenv
import os

from styles.page_style import set_page_config_centered, disable_header_and_footer


load_dotenv()

# Настройка страницы
set_page_config_centered()
disable_header_and_footer()

st.markdown(
    """
    # ETL BTC Dashboard
    
    Дашборд, позволяющий отследить информацию из хранилищ данных.
    
    - Данные курса и сигналова берутся из InfluxDB;
    - Новостные данные берутся из Qdrant
    """
)
