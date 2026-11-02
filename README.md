GitHub Repo Star Regressor.
An interactive Streamlit app that predicts star count for repositories using a tuned scikit-learn Random Forest Regressor.
Users can choose between two functionalities: finding predictions on the top 100 weekly repos displayed in a table, with hyperlinks to each repo in the table. The user can also make predictions on their chosen repo by adding the creator and name of the repo or by pasting a direct link to the repo into the text boxes. 

Key features:
• Predictions on top 100 weekly GitHub repositories.
• Predictions on a user specified repository. 
• View current features of user specified repository fetched directly from GitHub.
• Interactive result table with hyperlinks to displayed repositories.
• Custom styled deployed Streamlit web interface.
• Live data fetching from GitHub API
• About page for brief description of application. 
• ML model caching for optimisation.

What files makes up the application:
• requirements.txt to detail the libraries needed for the programs deployment via streamlit.
• A .pkl file containing the model that makes predictions.
• .csv files containing data used for training models. 
Python files:
• A data extraction file that fetches a user specified amount of real GitHub repositories from the GitHub API.
• A data inspection file using pandas DataFrame operations to see important details of the extracted data, such as the data type and number of null values for each feature. 
• A data cleaning file that removes features not used in the training of the model, creates and alters derived features for optimal training and splits the data into training and testing sets.
• An evaluation file that creates, trains and tests a baseline pipeline (pre-processor + model). It includes a function that creates learning curve graphs to help analyse overfitting and underfitting during model development by comparing training and testing scores for different amounts of data. Another function creates a scatter graph displaying how the baseline and trained model fit the training data.
• A tune file that runs a Grid Search Cross Validation that trains multiple models on subsets of the training data and validates them using the remaining fold it wasn't trained on. It returns the model that had the highest validation score along with the hyperparameters that did best. My GSCV contained 5 combinations of single or double stacked models per run with a range of hyperparameters totalling roughly 1500 candidate models per run. This was repeated as I swapped out models and tweaked hyperparameter values in order to find the best model for this problem. In total over 5000 different candidate models were trialed to get the final model. 
• A main file that combines the functionality all the previously mentioned files to run the full process and find the best model.
• An API file that fetches live repository features from GitHub API. One of the functions fetches a single repository by taking the repo name and creators name to fetch the features the model needs to make predictions. The other functions fetches the top 100 repos created in the last week to date, sorted by star count. 
• A predictor file that uses the model to predict the star count of a single repository or make a batch prediction on the 100 repositories passed from the API file when the user needs. It does this by creating a pandas DataFrame from the dictionary of a repos features and passing it to the model, fetched from the model path, to make a prediction. 
• An App file that creates the web interface using Streamlit to allow users to use the model on their chosen repos or discover the top 100 weekly repos. It contains two columns fitted with containers for the two features of the application. It takes user input via text boxes and buttons and returns predictions in the form of st.success boxes or a table containing the results. All repos returned are also in the form of hyperlinks to allow the user to visit the repository directly if they are interested. The webpage is styled using inline CSS added via an st.markdown, or using pandas styling for the tables to keep the web pages theme consistent. 

The model:
The best model found by the discussed GridSearchCV was a Random Forest Regressor with the following hyperparameters:
n_estimators: 50
max_depth: 20
random_state: 42
criterion: squared_error
bootstrap: True
min_samples_leaf: 1
min_samples_split: 2

This model achieves an R² score of 0.7889 on the test set, not GSCV validation fold, maintaining a consistent performance that shows the model didn't overfit the training data. The baseline Ridge model had an R² score of around 0.3. This is an impressive improvement and score that shows how the model is able to capture the complex patterns of the problem and the correlations between a repos features and its star count. I believe the remaining 0.2111 is due to the large amount of noise in the data and the unpredictability of repositories success, there are many important details that a ML model struggles to capture by just seeing certain features extracted from the GitHub API. For example, if an owner of a repo is famous in the tech space or the repo has benefitted from success on social media. My model found a 68% importance leaning towards the number of forks a repo has. This was a wide margin compared to the second most important feature and shows that a the community adoption of a repository is very important, which makes sense as stars would be more likely to communicate if developers are building upon the repository. The second most important feature was the size of the repository which shows that complex and feature-complete repositories gain more attention that more less detailed repos. Other features that ranked lower are the number of open issues, which shows users tracking bugs in the repo meaning it is getting used and the developer gets feedback and can make improvements. The age and programming language showed a low correlation to the success of the repository as they don't indicate active engagement. 

Visit the website at: https://repository-star-predictor.streamlit.app/

 
Installation:
git clone https://github.com/GeorgeAJC/github_star_predictor.git
cd github_star_predictor
pip install -r requirements.txt
python -m streamlit run app.py
