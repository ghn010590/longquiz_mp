import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from diabetes_model.config.core import config
from diabetes_model.processing.features import Mapper


preprocessor1 = ColumnTransformer(
    transformers=[
        # Individual numerical columns
        ('scale_age', StandardScaler(), config.model_config.age_var),
        ('scale_bmi', StandardScaler(), config.model_config.bmi_var),
        ('scale_HbA1c', StandardScaler(), config.model_config.HbA1c_level_var),
        ('scale_blood_glucose', StandardScaler(), config.model_config.blood_glucose_level_var),
        # Individual categorical columns
        ('impute_gender', SimpleImputer(strategy='most_frequent'), ['gender']),
        ('impute_smoking_history', SimpleImputer(strategy='most_frequent'), ['smoking_history'])
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('scale_age', StandardScaler(), [config.model_config.age_var]),
        ('scale_bmi', StandardScaler(), [config.model_config.bmi_var]),
        ('scale_HbA1c', StandardScaler(), [config.model_config.HbA1c_level_var]),
        ('scale_blood_glucose', StandardScaler(), [config.model_config.blood_glucose_level_var]),
        #('impute_gender', SimpleImputer(strategy='most_frequent'), ['gender']),
        #('impute_smoking_history', SimpleImputer(strategy='most_frequent'), ['smoking_history'])
    ],
    remainder='passthrough'  # To keep other columns not listed here
)



diabetes_pipe1 = Pipeline([

    
    ('map_smoking_history', Mapper(variable = config.model_config.smoking_history_var, mappings = config.model_config.smoking_history_mappings)),
    ('map_gender', Mapper(variable = config.model_config.smoking_history_var, mappings = config.model_config.smoking_history_mappings)),
    ('preprocessor', preprocessor),
    
    # Regressor
    ('model_xgb', XGBClassifier( n_estimators = 100,learning_rate=0.1,max_depth=6,min_child_weight=1,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic'))
    
    ])



diabetes_pipe = Pipeline([
    ('map_gender', Mapper(variable=config.model_config.gender_var, mappings=config.model_config.gender_mappings)),
    ('map_smoking_history', Mapper(variable=config.model_config.smoking_history_var, mappings=config.model_config.smoking_history_mappings)),
    ('preprocessor', preprocessor),
    ('model_xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic'))
])
