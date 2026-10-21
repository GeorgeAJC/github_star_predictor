from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
from xgboost import XGBRegressor
import numpy as np

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

    full_pipeline = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p, inverse_func=np.expm1)
    

    param_grid = [  
        {
            'regressor__preprocessor__num__scaler': [StandardScaler()],
            'regressor__preprocessor__cat__encoder': [OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],'regressor__model': [GradientBoostingRegressor(random_state=42)],
            'regressor__model__n_estimators': [200, 400, 600],
            'regressor__model__learning_rate': [0.02, 0.04],
            'regressor__model__max_depth': [3, 5],
            'regressor__model__subsample': [0.8, 1.0],
            'regressor__model__min_samples_leaf': [20, 30, 50],
        },
        {
            'regressor__preprocessor__num__scaler': [StandardScaler()],
            'regressor__preprocessor__cat__encoder': [OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],'regressor__model': [RandomForestRegressor(random_state=42)],
            'regressor__model__n_estimators': [200, 400, 600],
            'regressor__model__max_depth': [3, 5, 10],
            'regressor__model__min_samples_split': [10, 15, 20],
            'regressor__model__min_samples_leaf': [10, 20, 30]
        },
        {
            'regressor__preprocessor__num__scaler': [StandardScaler()],
            'regressor__preprocessor__cat__encoder': [OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first') 
            ],'regressor__model': [StackingRegressor(
                estimators=[
                    ('ridge', Ridge()), 
                    ('svr', SVR())
                ],final_estimator=Ridge()
            )],
            'regressor__model__ridge__alpha': [10.0, 50.0],
            'regressor__model__svr__C': [0.1, 1.0],
            'regressor__model__final_estimator__alpha': [1.0, 5.0]
        },
        {
            'regressor__preprocessor__num__scaler': [StandardScaler()],
            'regressor__preprocessor__cat__encoder': [OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],
            'regressor__model': [StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor(random_state=42)), 
                    ('knn', KNeighborsRegressor())
                ],final_estimator=GradientBoostingRegressor(random_state=42)
            )],'regressor__model__rf__n_estimators': [200, 400],
            'regressor__model__knn__n_neighbors': [5, 10, 25],
            'regressor__model__final_estimator__n_estimators': [200, 400]
        },
        {
            'regressor__preprocessor__num__scaler': [StandardScaler()],
            'regressor__preprocessor__cat__encoder': [OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            ],
            'regressor__model': [XGBRegressor(random_state=42)],
            'regressor__model__n_estimators': [200, 400],
            'regressor__model__colsample_bytree': [0.6,0.8, 1.0],
            'regressor__model__max_depth': [3, 5],
            'regressor__model__subsample': [0.8, 1.0],
            'regressor__model__learning_rate': [0.05, 0.1],       
            'regressor__model__reg_alpha': [0.5, 1.0],      
            'regressor__model__reg_lambda': [1.0, 5.0, 10.0],
            'regressor__model__reg_gamma': [1.0, 5.0]
        }
        
    ]
    
    grid_search_cv = GridSearchCV(
        estimator=full_pipeline,
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