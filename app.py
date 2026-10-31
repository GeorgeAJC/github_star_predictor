import streamlit as st
import pandas as pd
from src.api import fetch_repo_features, fetch_top_100_new_weekly_repos
from src.predictor import load_model, predict_single_repo_stars, predict_multiple_repos_stars

st.set_page_config(page_title="GitHub Star Predictor", page_icon=":star:", layout="wide")


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
    with col_right:
        with st.container(border=True):
            st.subheader("Single Repository Lookup")
            st.markdown("Evaluate any specific public GitHub repository.")
            
            owner = st.text_input("GitHub Owner / Username", "torvalds")
            repo = st.text_input("Repository Name", "linux")

            if st.button("Predict Stars for Repo", type="primary"):
                if not owner or not repo:
                    st.warning("Please enter both an owner and a repository name.")
                else:
                    with st.spinner(f"Fetching data for {owner}/{repo} from GitHub..."):
                        repo_data = fetch_repo_features(owner, repo)
                    
                    if repo_data:
                        if isinstance(repo_data, str) and repo_data.startswith("Error"):
                            st.error(repo_data)
                        else:
                            model_input = {k: v for k, v in repo_data.items() if k not in ["repo_name"]}
                            predicted_stars = predict_single_repo_stars(model, model_input)
                            
                            st.success(f"Repository: {owner}/{repo}")
                            
                            st.metric(label="Predicted Stars:", value=predicted_stars)

                            
                            with st.expander("View Extracted Features"):
                                st.json(model_input)
                    else:
                        st.error("Could not find that repository. Check spelling or rate limits.")

elif selected_page == "About":
    st.title("About This App")
    st.markdown("""
        Welcome to the GitHub Star Predictor!
        This program uses machine learning models to analyze public GitHub repositories and predict their future star trajectories based off features fetched directly from GitHub using it's API.
        
        * **Built with:** Python, Streamlit for web interface, and Scikit-Learn for ML model.
        * **Designed by:** George Churchill! 
    """)