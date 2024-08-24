"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from diabetes_model.predict import make_prediction

def test_make_prediction(sample_input_data):
 
    X_test, y_test = sample_input_data
    expected_num_of_predictions = len(X_test)

 
    result = make_prediction(input_data=X_test)

 
    predictions = result.get("predictions")
    assert isinstance(predictions, list)  
    assert result.get("errors") is None
    assert len(predictions) == expected_num_of_predictions
    

    predictions = np.array(predictions)

    # Calculating performance metrics
    f1 = f1_score(y_test, predictions, average='binary')
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    
    assert f1 > 0.8
    assert precision > 0.95
    assert recall > 0.6
