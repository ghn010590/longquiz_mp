from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
diabetes_dataset = pd.read_csv("diabetes_prediction_dataset.csv") 
diabetes_dataset['gender'].replace(['Other', 'Male', 'Female'], [0, 1, 2], inplace=True)
diabetes_dataset['smoking_history'].replace(['never', 'No Info', 'current', 'former', 'ever', 'not current'], [0, 1, 2, 3, 4, 5], inplace=True)

# Filter out 'Other' gender as the values were very less
diabetes_dataset = diabetes_dataset[diabetes_dataset['gender'] != 0]
diabetes_dataset = diabetes_dataset.reset_index(drop=True)

# Identify categorical and numerical columns
categorical_cols = ['gender', 'smoking_history']
numerical_cols = diabetes_dataset.columns.difference(['diabetes'] + categorical_cols).tolist()

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)
    ])

# Define the pipeline with hyperparameters
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic'
    ))
])

# Split the data
X = diabetes_dataset.drop("diabetes", axis=1)
y = diabetes_dataset["diabetes"]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the pipeline
pipeline.fit(X_train, Y_train)

# Predictions
Y_pred = pipeline.predict(X_test)

# Evaluation
train_accuracy = pipeline.score(X_train, Y_train)
test_accuracy = pipeline.score(X_test, Y_test)
f1 = f1_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)

print('Training accuracy: {:.4f}'.format(train_accuracy))
print('Testing accuracy: {:.4f}'.format(test_accuracy))
print('F1 Score: {:.4f}'.format(f1))
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
