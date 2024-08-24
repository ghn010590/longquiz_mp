import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from diabetes_model import __version__ as _version
from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_pipeline
#from diabetes_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
diabetes_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""
    
    # Ensure input_data is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])  # Convert dict to DataFrame

    # Make predictions using the loaded pipeline
    predictions = diabetes_pipe.predict(input_data)

    # Floor predictions and convert to integers
    predictions = np.floor(predictions).astype(int)

    # Prepare results including all predictions
    results = {
        "predictions": predictions.tolist(),  # Convert numpy array to list for better readability
        "version": _version
    }

    # Print results in a human-readable way
    print(f"Predictions: {results['predictions']}")
    print(f"Model version: {results['version']}")

    return results

if __name__ == "__main__":
    # Example input data for a single sample
    data_in = {
        'gender': "Male",
        'age': 50,
        'hypertension': 1,
        'heart_disease': 0,
        'smoking_history': "current",
        'bmi': 27.32,
        'HbA1c_level': 5.7,
        'blood_glucose_level': 260
    }

    # Call the prediction function with the sample data
    make_prediction(input_data=data_in)