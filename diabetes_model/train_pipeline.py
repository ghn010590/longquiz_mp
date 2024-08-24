import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from diabetes_model.config.core import config
from diabetes_model.pipeline import diabetes_pipe
from diabetes_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    print(config.app_config.training_data_file)
    #data = load_dataset(file_name = config.app_config.training_data_file)
    data = pd.read_csv("C:\AI&MLOps\LongQuiz_MP_M3M4\diabetes_model\datasets\diabetes_prediction_dataset.csv") 
    
    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]
    
    # divide train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        
        X,     
        y,       
        test_size = 0.3,
        random_state= 42  
    )

    # Pipeline fitting
    diabetes_pipe.fit(X_train, Y_train)
    Y_pred = diabetes_pipe.predict(X_test)

    # Evaluation
    train_accuracy = diabetes_pipe.score(X_train, Y_train)
    test_accuracy = diabetes_pipe.score(X_test, Y_test)
    f1 = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)

    print('Training accuracy: {:.4f}'.format(train_accuracy))
    print('Testing accuracy: {:.4f}'.format(test_accuracy))
    print('F1 Score: {:.4f}'.format(f1))
    print('Precision: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    #y_pred = bikeshare_pipe.predict(X_test)

    # Calculate the score/error
    #print("R2 score:", r2_score(y_test, y_pred).round(2))
    #print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist = diabetes_pipe)
    
if __name__ == "__main__":
    run_training()