import pandas as pd

from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    #Drop repo names
    df = df.iloc[:, 1:]

    # Convert 'created_at' to repository age in days
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['age_days'] = (pd.Timestamp.now() - df['created_at'].dt.tz_localize(None)).dt.days
    df = df.drop(columns=['created_at'])

    #Separate features and target
    X = df.drop(columns=['stars'])
    y = df['stars']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    #split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

#test
if __name__ == "__main__":
    filepath = "C:\\programming\\CV PROJECTS\\github_star_predictor\\data\\github_repos_dataset.csv"
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_clean_data(filepath)
    print(X_train.info())
    print(X_test.info())
    print(y_train.info())
    print(y_test.info())
    print(numeric_features)
    print(categorical_features)
    print(type(categorical_features[0]))