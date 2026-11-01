import streamlit as st
import pandas as pd
from src.api import fetch_repo_features, fetch_top_100_new_weekly_repos
from src.predictor import load_model, predict_single_repo_stars, predict_multiple_repos_stars
import re

st.set_page_config(page_title="GitHub Repo Star Predictor", page_icon=":star:", layout="wide")

#inline styling using CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #F5F3FF;
        color: #1E1B4B;
    }

    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span {
        color: #1E1B4B !important;
        font-weight: 600 !important;
    }

    [data-testid="stRadio"] input[type="radio"] {
        accent-color: #6D5BD0 !important;
    }

    [data-testid="stRadio"] div[role="radiogroup"] {
        background-color: #DDD6FE;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #6D5BD0;
    }

    [data-testid="stColumn"] {
        background-color: #DDD6FE;
        border-right: 2px solid #6D5BD0;
        color: #18181B;
    }

    [data-testid="stContainer"] {
        min-height: 800px !important;
        height: auto !important;
        width: auto !important;
    }

    [data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    [data-testid="stButton"] button[kind="primary"] {
        background-color: #F59E0B;
        color: black;
        border: none;
    }
    
    [data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #D97706;
    }

    [data-testid="stDataFrame"], 
    [data-testid="stDataFrame"] > div {
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-x: auto !important;
        background-color: #DDD6FE !important;
        border: 1px solid #6D5BD0 !important;
        border-radius: 8px;
        padding: 10px;
    }

    [data-testid="stDataFrame"] iframe, 
    [data-testid="stDataFrame"] div[data-baseweb="card"] {
        background-color: #DDD6FE !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    [data-testid="stDataFrame"] th, 
    [data-testid="stDataFrame"] td {
        color: #1E1B4B !important;
        background-color: #DDD6FE !important;
    }

    [data-testid="stMetric"] {
        background-color: #DDD6FE;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #6D5BD0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {
        color: #1E1B4B !important; 
    }

    [data-testid="stTextInput"] input {
        border-radius: 6px;
        border: 1px solid #6D5BD0;
        background-color: #F5F3FF;
        color: #1E1B4B;
    }

    [data-testid="stTextInput"] label {
        color: #1E1B4B;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: #ff4b4b;
        box-shadow: 0 0 0 1px #DDD6FE;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details {
        border: 1px solid #6D5BD0;
        border-radius: 8px;
        background-color: #DDD6FE;
    }

    [data-testid="stAlert"],
    [data-testid="stAlert"] p {
        background-color: #6D5BD0;
        border: 1px solid #1E1B4B;
        color: #1E1B4B !important;
    }
    </style>
""", unsafe_allow_html=True)

#create radio buttons for navigation between home and about page
selected_page = st.radio(
    "Navigation", 
    ["Home", "About"], 
    horizontal=True, 
    label_visibility="collapsed" # Hides the word "Navigation" for a sleek look
)

#create 'Home' page contents
if selected_page == "Home":
    st.title("GitHub Repo Star Predictor")

    #load model and cache it so the model is not reloaded on each interaction. 
    @st.cache_resource
    def fetch_model():
        return load_model()

    #load model with a spinner to indicate loading
    with st.spinner("Loading machine learning model..."):
        model = fetch_model()

    #create columns for layout of two home page sections;
    col_left, col_right = st.columns(2, gap="small")

    #left column for batch repo predictions
    with col_left:
            #create container for batch repo predictions
            with st.container(border=True):
                st.subheader("Top 100 New Repos (This Week)")
                st.markdown("Query trending repositories created over the past 7 days and run batch predictions.")

                #button to start fetch and prediction
                if st.button("Fetch & Predict Top 100", type="primary"):
                    #fetch data with spinner while loading
                    with st.spinner("Querying GitHub Search API and running batch inference..."):
                        recent_repos = fetch_top_100_new_weekly_repos()

                    #check if recent_repos returned None
                    if recent_repos:
                        #if an error returned, show error message, else run preductions
                        if isinstance(recent_repos, str) and recent_repos.startswith("Error"):
                            st.error(recent_repos)
                        else:
                            #remove repo_name and actual starts, add every other key, value pair of features to dictionary
                            model_inputs = [{k: v for k, v in r.items() if k not in ["repo_name", "actual_stars"]} for r in recent_repos]
                            predicted_stars_list = predict_multiple_repos_stars(model, model_inputs) #predict

                            #convert to dataframe
                            results = []
                            for i, r in enumerate(recent_repos):
                                results.append({
                                    "Repository": r["repo_name"],
                                    "Language": r["language"],
                                    "Predicted": predicted_stars_list[i],
                                })

                            results_df = pd.DataFrame(results)

                            #make repo names urls in order to make them clickable in the dataframe
                            results_df["Repository"] = "https://github.com/" + results_df["Repository"]

                            #style df to appearnace of website
                            styled_results_df = results_df.style.set_properties(**{
                                'background-color':'#DDD6FE',
                                'color': '#1E1B4B',
                                'border-color': '#6D5BD0'
                            })
                            st.dataframe(styled_results_df,
                                        width='stretch',
                                        hide_index=True,
                                        column_config={
                                            "Repository": st.column_config.LinkColumn(
                                                label="Repository",
                                                display_text=r"https://github.com/(.+)$",  # Regex to extract the repo name from the URL
                                                help="Click to visit the repository on GitHub")
                                        }
                            )
                    else:
                        st.warning("Could not fetch recent repositories. Please try again shortly.")
    #right column for singhle repo predictions
    with col_right:
        with st.container(border=True):
            st.subheader("Single Repository Lookup")
            st.markdown("Evaluate any specific public GitHub repository." \
            "Enter the owner and repository name, or a full GitHub repository URL to fetch features and predict stars." \
            "If you enter both a URL and owner/repo, the URL will be used.")

            #add text inputs for owner username and repo name and default value.
            owner = st.text_input("GitHub Owner / Username", "GeorgeAJC")
            repo = st.text_input("Repository Name", "github_star_predictor")
            #add text input so a url can be inputted
            repo_url = st.text_input("Or enter a full GitHub repository URL", "")

            if st.button("Predict Stars for Repo", type="primary"):
                #error if both owner and repo arent entered
                if not(owner and repo) and not repo_url:
                    st.warning("Please enter both an owner and a repository name, or a full GitHub repository URL.")
                else:
                    with st.spinner(f"Fetching data for {owner}/{repo} from GitHub..."):
                        #get repo name and owner from url if entered
                        if repo_url:
                            match = re.search(r"github.com/([^/]+)/([^/]+)$", repo_url)
                            if match:
                                owner = match.group(1)
                                repo = match.group(2)
                                repo_data = fetch_repo_features(owner, repo)   
                            else:
                                st.error("Invalid GitHub repository URL. Please enter a valid URL.")
                                repo_data = None
                        elif owner and repo:
                            repo_data = fetch_repo_features(owner, repo)
                        else:
                            st.error("Please enter both a valid owner and a repository name, or a valid full GitHub repository URL.")
                            repo_data = None
                    
                    if repo_data:
                        #error handling as in batch repo predictions
                        if isinstance(repo_data, str) and repo_data.startswith("Error"):
                            st.error(repo_data)
                        else:
                            model_input = {k: v for k, v in repo_data.items() if k not in ["repo_name"]}
                            predicted_stars = predict_single_repo_stars(model, model_input)

                            #show repo fetched in success message
                            repo_name = f"{owner}/{repo}"
                            repo_url = f"https://github.com/{repo_name}"
                            st.success(f"Repository: [{repo_name}]({repo_url})")

                            #show prediction
                            st.metric(label="Predicted Stars:", value=predicted_stars)

                            #show the extracted features in expander
                            with st.expander("View Extracted Features"):
                                st.json(model_input)
                    else:
                        st.error("Could not find that repository. Check spelling or rate limits.")

#about page, title and text
elif selected_page == "About":
    st.title("About This App")
    st.markdown("""
        Welcome to the GitHub Star Predictor!
        This program uses machine learning models to analyze public GitHub repositories and predict their future star trajectories based off features fetched directly from GitHub using it's API.
        20,000 real repos were fetched along with their features such as language, size, forks and more in order to train the model.
        Over 5000 combinations of single and stacked models with different hyperparamters were Cross Validated in order to find the final model.
        It achieves an impressive 0.7889 r2 score, capturing the underlying patterns behind what drives a repositories star count and sucess. The
        remaining 0.21 can be explained as a result of the external noise and the fact that some variables affecting a repos success are 
        uncontrollable and unpredictable. For example, social media attention.

        * **Built with:** Python, Streamlit for web interface, and Scikit-Learn for ML model.
        * **Designed by:** George Churchill! 
    """)