import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pytest import fixture

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


from diabetes_model.processing.features import Mapper


@fixture
def sample_input_data():
    data = {
        'gender': ['Male', 'Female', np.nan, 'Female'],
        'smoking_history': ['never', 'former', 'current', np.nan],
        'age': [25, 35, 45, 55]  
    }
    return pd.DataFrame(data)


gender_mappings = {'Male': 1, 'Female': 0}
smoking_history_mappings = {'never': 0, 'former': 1, 'current': 2, np.nan: -1}

def test_gender_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable='gender', mappings=gender_mappings)
    
    # When
    subject = mapper.fit_transform(sample_input_data)
    
    # Then
    assert subject['gender'].tolist() == [1, 0, -1, 0]  

def test_smoking_history_variable_mapper(sample_input_data):
    mapper = Mapper(variable='smoking_history', mappings=smoking_history_mappings)
    subject = mapper.fit_transform(sample_input_data)
    assert subject['smoking_history'].tolist() == [0, 1, 2, -1]  

