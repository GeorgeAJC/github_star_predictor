import joblib
import pandas as pd
import pandas as pd
#imports for tests
from src.api import fetch_repo_features, fetch_top_100_new_weekly_repos

def load_model(model_path):
    return joblib.load(model_path)


def predict_single_repo_stars(model, repo_data):

    df = pd.DataFrame([repo_data])

    prediction = model.predict(df)

    return int(round(prediction[0]))

def predict_multiple_repos_stars(model, repos_data):
    
    df = pd.DataFrame(repos_data)

    predictions = model.predict(df)

    return [int(round(pred)) for pred in predictions]

#testing
if __name__ == "__main__":
    model_path = "models/trained_model_0.7889.pkl"
    model = load_model(model_path)

    #test single repo prediction
    owner = "GeorgeAJC"
    repo = "github_star_predictor"
    repo_data = fetch_repo_features(owner, repo)
    predicted_stars = predict_single_repo_stars(model, repo_data)
    print(f"Predicted stars: {predicted_stars}")


    #test multiple repos prediction
    multiple_repo_data = fetch_top_100_new_weekly_repos()
    predicted_stars_list = predict_multiple_repos_stars(model, multiple_repo_data)
    for repo_info, predicted_stars in zip(multiple_repo_data, predicted_stars_list):
        print(f"Predicted stars for {repo_info['repo_name']}: {predicted_stars}")
