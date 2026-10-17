import os
import joblib
from src.tune import tune_hyperparameters
from src.evaluate import run_full_evaluation
from src.data_cleaning import load_and_clean_data

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_clean_data("data/github_repos_dataset.csv")

    model_path = "models/best_pipeline.pkl"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        best_model = joblib.load(model_path)
    else:
        print("Training new model via GridSearchCV...")
        best_model = tune_hyperparameters(X_train, y_train, numeric_features, categorical_features)
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

    run_full_evaluation(
        best_model, X_train, y_train, X_test, y_test, numeric_features, categorical_features
    )

if __name__ == "__main__":
    main()
 