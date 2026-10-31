import os
import joblib
from src.tune import tune_hyperparameters
from src.evaluate import run_full_evaluation
from src.data_cleaning import load_and_clean_data

def main(retrain=False):
    #load all data and prepare for training
    print("Loading data...")
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_clean_data("data/github_repos_dataset_20k.csv")

    #get path for best model found
    best_model_path = "models/best_pipeline.pkl"
    os.makedirs("models", exist_ok=True)

    #use best saved model if it exists
    if os.path.exists(best_model_path) and not retrain:
        print(f"Loading model from {best_model_path}...")
        trained_model = joblib.load(best_model_path)
    else:
        #if no model found, run new GSCV to find best model on training data and save it
        print("Training new model via GridSearchCV...")
        trained_model = tune_hyperparameters(X_train, y_train, numeric_features, categorical_features)
        trained_model_path = f"models/trained_model_{trained_model.score(X_test, y_test):.4f}.pkl"
        joblib.dump(trained_model, trained_model_path)
        print(f"Model saved to {trained_model_path}")

    #evaluate model to compare to baseline and get learning cruve and scatter comparison graphs 
    run_full_evaluation(
        trained_model, X_train, y_train, X_test, y_test, numeric_features, categorical_features
    )
#run
if __name__ == "__main__":
    main(True)
 