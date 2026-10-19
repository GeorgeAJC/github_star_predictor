from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings

def tune_hyperparameters(X_train, y_train, numeric_features, categorical_features):

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    #baseline for pipeline, will be replaced by GridSearchCV
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge()) 
    ])
    

    param_grid = [  
        {
            'preprocessor__num__scaler': [StandardScaler()],
            'preprocessor__cat__encoder': [
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],'model': [GradientBoostingRegressor(random_state=42)],
            'model__n_estimators': [500, 1000],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 6]
        },
        {
            'preprocessor__num__scaler': [StandardScaler()],
            'preprocessor__cat__encoder': [
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],'model': [RandomForestRegressor(random_state=42)],
            'model__n_estimators': [500, 1000],
            'model__max_depth': [10, 20],
            'model__min_samples_split': [10, 50],
            'model__min_samples_leaf': [5, 20]
        },
        {
            'preprocessor__num__scaler': [StandardScaler()],
            'preprocessor__cat__encoder': [
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first') 
            ],'model': [StackingRegressor(
                estimators=[
                    ('ridge', Ridge()), 
                    ('svr', SVR())
                ],final_estimator=Ridge()
            )],
            'model__ridge__alpha': [1.0, 10.0],
            'model__svr__C': [10.0, 100.0],
            'model__epsilon': [0.01, 0.1, 0.2],
            'model__final_estimator__alpha': [1.0, 10.0]
        },
        {
            'preprocessor__num__scaler': [StandardScaler()],
            'preprocessor__cat__encoder': [
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],
            'model': [StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(random_state=42)), 
                    ('knn', KNeighborsRegressor())
                ],final_estimator=GradientBoostingRegressor(random_state=42)
            )],'model__rf__n_estimators': [500, 1000],
            'model__knn__n_neighbors': [5, 9, 15],
            'model__final_estimator__n_estimators': [500, 1000]
        },
        {
            'preprocessor__num__scaler': [StandardScaler()],
            'preprocessor__cat__encoder': [
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],
            'model': [StackingRegressor(
                estimators=[
                    ('inner_stack', StackingRegressor(
                        estimators=[('ridge', Ridge()), ('svr', SVR())],
                        final_estimator=RandomForestRegressor(random_state=42)
                    )),
                    ('knn', KNeighborsRegressor())
                ],final_estimator=Lasso()
            )],'model__inner_stack__ridge__alpha': [10.0],
            'model__inner_stack__final_estimator__n_estimators': [500],
            'model__knn__n_neighbors': [5, 9, 15],
            'model__final_estimator__alpha': [1.0, 10.0]
        }
    ]
    
    grid_search_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    warnings.filterwarnings("ignore", message="Found unknown categories")
    
    print("Running GridSearchCV...")
    grid_search_cv.fit(X_train, y_train)
    print("\nGridSearchCV finished.")
    print(f"Best r2 score: {grid_search_cv.best_score_}")
    print(f"Best parameters:\n{grid_search_cv.best_params_}")
    
    return grid_search_cv.best_estimator_