import streamlit as st
import pandas as pd
from src.api import fetch_repo_features, fetch_top_100_new_weekly_repos
from src.predictor import load_model, predict_single_repo_stars, predict_multiple_repos_stars