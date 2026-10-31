import os
import time
import requests
import pandas as pd

#building datasets for training
def fetch_github_dataset(target_count=20000):
    url = "https://api.github.com/search/repositories"
    headers = {"Accept": "application/vnd.github+json"}
    
    repos = []
    # Sliced into monthly time frames to bypass the 1,000-result search limit
    date_ranges = pd.date_range(start="2023-01-01", end="2026-01-01", freq="1ME")
    
    print(f"Fetching data")
    
    for i in range(len(date_ranges) - 1):
        if len(repos) >= target_count:
            break

        #convert date formats
        start_date = date_ranges[i].strftime("%Y-%m-%d")
        end_date = date_ranges[i+1].strftime("%Y-%m-%d")
        
        page = 1
        while page <= 20:  # Max 20 pages per query (1,000 results per time slice totalling 20,000 results)
            query = f"created:{start_date}..{end_date} stars:>5"
            params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
            
            response = requests.get(url, headers=headers, params=params)

            #deal with API errors
            if response.status_code == 403:  # Rate limit hit
                print("Rate limit reached. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
            elif response.status_code == 422: # Validation limit reached
                break
            elif response.status_code != 200:
                print(f"Error {response.status_code}: {response.json()}")
                break
                
            items = response.json().get("items", [])
            #stop the loop if response is empty despite no API errors
            if not items:
                break

            #create dictionary for each repo and append to array of repos 
            for item in items:
                repos.append({
                    "repo_name": item["full_name"],
                    "created_at": item["created_at"],
                    "stars": item["stargazers_count"],
                    "forks": item["forks_count"],
                    "open_issues": item["open_issues_count"],
                    "size": item["size"],
                    "language": item["language"],
                })

            #stop loop if target count of repos is met    
            if len(repos) >= target_count:
                break
                
            page += 1

    #convert array to pandas dataframe, then to CSV and save to models/ with a descriptive name
    df = pd.DataFrame(repos)
    if not df.empty:
        df = df.head(target_count)
        os.makedirs("data", exist_ok=True)
        file_path = "data/github_repos_dataset_20k.csv"
        df.to_csv(file_path, index=False)
        print(f"Successfully fetched and saved data to {file_path}")
    else:
        print("Repository data fetch failed.")
    return df

#run function to create test data
if __name__ == "__main__":
    fetch_github_dataset(target_count=20000)