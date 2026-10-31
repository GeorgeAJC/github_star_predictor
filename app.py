import streamlit as st
import pandas as pd
from src.api import fetch_repo_features, fetch_top_100_new_weekly_repos
from src.predictor import load_model, predict_single_repo_stars, predict_multiple_repos_stars

selected_page = st.radio(
    "Navigation", 
    ["Home", "About"], 
    horizontal=True, 
    label_visibility="collapsed" # Hides the word "Navigation" for a sleek look
)

if selected_page == "Home":
    st.title("GitHub Star Predictor")

    @st.cache_resource
    def fetch_model():
        return load_model()

    with st.spinner("Loading machine learning model..."):
        model = fetch_model()

    col_left, col_right = st.columns(2, gap="small")

    with col_left: