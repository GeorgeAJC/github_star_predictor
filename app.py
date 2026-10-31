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
            with st.container(border=True):
                st.subheader("Top 100 New Repos (This Week)")
                st.markdown("Query trending repositories created over the past 7 days and run batch predictions.")

                if st.button("Fetch & Predict Top 100", type="primary"):
                    with st.spinner("Querying GitHub Search API and running batch inference..."):
                        recent_repos = fetch_top_100_new_weekly_repos()
                        
                    if recent_repos:
                        if isinstance(recent_repos, str) and recent_repos.startswith("Error"):
                            st.error(recent_repos)
                        else:
                            model_inputs = [{k: v for k, v in r.items() if k not in ["repo_name", "actual_stars"]} for r in recent_repos]
                            predicted_stars_list = predict_multiple_repos_stars(model, model_inputs)
                            
                            results = []
                            for i, r in enumerate(recent_repos):
                                results.append({
                                    "Repository": r["repo_name"],
                                    "Language": r["language"],
                                    "Predicted": predicted_stars_list[i],
                                })

                            results_df = pd.DataFrame(results)
                            styled_results_df = results_df.style.set_properties(**{
                                'background-color':'#DDD6FE',
                                'color': '#1E1B4B',
                                'border-color': '#6D5BD0'
                            })
                            st.dataframe(styled_results_df, width='stretch', hide_index=True)
                            
                    else:
                        st.warning("Could not fetch recent repositories. Please try again shortly.")