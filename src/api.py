from datetime import datetime, timedelta
from wsgiref import headers
import requests

#fetching details used in single repo predictor
def fetch_repo_features(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}" #build url
    response = requests.get(url)
    
    if response.status_code == 200: #check that request doesn't return an error
        data = response.json()
    else:
        return "Error fetching repo, error code: " + str(response.status_code)

    creation_date = datetime.strptime(data["created_at"], "%Y-%m-%dT%H:%M:%SZ") #build dervied feature
    age_days = (datetime.now() - creation_date).days

    #return features as a dictionary 
    return{
        "repo_name": data["name"],
        "age_days": age_days,
        "forks": data["forks_count"],
        "open_issues": data["open_issues_count"],
        "size": data["size"],
        "language": data["language"],
    }

#fetch batch details of multiple repos
def fetch_top_100_new_weekly_repos():
    url = "https://api.github.com/search/repositories"

    #create date that all repos must have been created after
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    #filter out tiny repos 
    star_threshold = 5

    
    query = f"created:>{one_week_ago} stars:>{star_threshold}"
    #sort repos via stars from most to least, only take 100 results
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100}
    
    response = requests.get(url, params=params)

    #if no error: loop through each repo, add feature to a dictionary and add to list of repos
    if response.status_code == 200:
        items = response.json().get("items", [])
        repos = []
        for item in items: #get features for each repo in the top 100 new weekly repos
            age_days = (datetime.now() - datetime.strptime(item["created_at"], "%Y-%m-%dT%H:%M:%SZ")).days
            repos.append({
                "repo_name": item["full_name"],
                "age_days": age_days,
                "stars": item["stargazers_count"],
                "forks": item["forks_count"],
                "open_issues": item["open_issues_count"],
                "size": item["size"],
                "language": item["language"],
            })
        return repos
    else:
        #if fetch is unsucessful return error code for more debugging
        return f"Error fetching top 100 new weekly repos, error code: {response.status_code}"

#testing
if __name__ == "__main__":
    owner = "GeorgeAJC"
    repo = "github_star_predictor"
    features = fetch_repo_features(owner, repo)
    print(features)

    top_repos = fetch_top_100_new_weekly_repos()
    print(top_repos)