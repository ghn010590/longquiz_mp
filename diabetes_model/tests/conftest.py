import sys
from pathlib import Path
import pytest
from sklearn.model_selection import train_test_split

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_dataset

@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config.training_data_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],     # predictors
        data[config.model_config.target],       # target
        test_size=0.3,
        random_state=42
    )
    return X_test, y_test
