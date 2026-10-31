import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
from src.data_cleaning import load_and_clean_data

#run a baseline model and provide a scatter plot comparison and learning curve for the baselinew and best model
def run_full_evaluation(best_model, X_train, y_train, X_test, y_test, numeric_features, categorical_features):
    
    print("\nRunning Baseline...")

    #simple preprocessor using simple imputer, standard scaler and one hot incoder for baseline
    baseline_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    #combine preprocessor pipeline with basic Ridge model as baseline
    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', baseline_preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    #train baseline on data and retrieve scores
    baseline_pipeline.fit(X_train, y_train)
    baseline_r2_score = baseline_pipeline.score(X_test, y_test)
    print(f"Baseline R2 Score: {baseline_r2_score:.4f}")


    #graphs
    plot_scatter_comparison(baseline_pipeline, best_model, X_test, y_test)
    plot_learning_curve(best_model, X_train, y_train, X_test, y_test)

#create scatter comparison graph to see how model fits data 
def plot_scatter_comparison(baseline_pipeline, best_model, X_test, y_test):

    print("Generating model visualisation...")

    #make predictions
    y_pred_baseline = baseline_pipeline.predict(X_test)
    y_pred_tuned = best_model.predict(X_test)

    #find min and max for reference line
    min_val = min(min(y_test), min(y_pred_baseline), min(y_pred_tuned))
    max_val = max(max(y_test), max(y_pred_baseline), max(y_pred_tuned))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    #baseline graph labels, plots and settings
    sns.scatterplot(x=y_test, y=y_pred_baseline, ax=axes[0], alpha=0.6, color='gray')
    axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    axes[0].set_title("Baseline Model (Ridge + Standard Scaler)")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].grid(True, alpha=0.3)
    #best model graph labels, plot and settings 
    sns.scatterplot(x=y_test, y=y_pred_tuned, ax=axes[1], alpha=0.7, color='royalblue')
    axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Tuned Model (Best Pipeline & Stack)")
    axes[1].set_xlabel("Actual Values")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Baseline vs. Tuned Model Performance", fontsize=14)
    plt.tight_layout()
    #save in models/ for future reference when comparing multiple models 
    plt.savefig(f"models/scatter_comparison_{best_model.score(X_test, y_test):.4f}.png") #use score as naming convention for easy differentiation
    print("Saved image in models/")
    plt.show()

#create method to display learning curve to 
def plot_learning_curve(model, X_train, y_train, X_test, y_test):

    print("Generating learning curve line graph...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        cv=5, 
        scoring='r2', 
        train_sizes=np.linspace(0.1, 1.0, 5), 
        n_jobs=-1
    )

    #calculate mean and standard deviation to add to graph
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(val_scores, axis=1)
    validation_std = np.std(val_scores, axis=1)

    #create graph, plot training and CV scores, add line spanning between mean +- std
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o') #line for training score
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.plot(train_sizes, validation_mean, label='Cross-Validation Score', color='darkorange', marker='o') #line for cross-validation score
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='darkorange', alpha=0.1)

    #labels and other settings for better looking graph
    plt.xlabel("Training Set Size")
    plt.ylabel("R2 Score")
    plt.title("Model Learning Curve (Overfitting & Generalization)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    #save in models/ for when comparing how overfit/underfit models are
    plt.savefig(f"models/learning_curve_{model.score(X_test, y_test):.4f}.png") #score as naming convention for easy differentiation
    print("Saved in models/")
    plt.show()

#test with makeshift pipeline
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = load_and_clean_data("data/github_repos_dataset.csv")
    #makeshift preprocessor for test
    test_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
    #makeshift pipeline (preprocessor +model) for test 
    test_pipeline = Pipeline(steps=[
        ('preprocessor', test_preprocessor),
        ('model', Ridge(alpha=1.0))
        ])
    test_pipeline.fit(X_train, y_train)
    run_full_evaluation(test_pipeline, X_train, y_train, X_test, y_test, numeric_features, categorical_features)
