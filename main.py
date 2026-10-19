import os
import joblib
from src.tune import tune_hyperparameters
from src.evaluate import run_full_evaluation
from src.data_cleaning import load_and_clean_data

def main(retrain=False):
    print("Loading data...")
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_clean_data("data/github_repos_dataset.csv")

    best_model_path = "models/best_pipeline.pkl"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(best_model_path) and not retrain:
        print(f"Loading model from {best_model_path}...")
        best_model = joblib.load(best_model_path)
    else:
        print("Training new model via GridSearchCV...")
        trained_model = tune_hyperparameters(X_train, y_train, numeric_features, categorical_features)
        trained_model_path = f"models/trained_model_{trained_model.score(X_test, y_test)}.pkl"
        joblib.dump(trained_model, trained_model_path)
        print(f"Model saved to {trained_model_path}")

    run_full_evaluation(
        best_model, X_train, y_train, X_test, y_test, numeric_features, categorical_features
    )

if __name__ == "__main__":
    main(True)
 