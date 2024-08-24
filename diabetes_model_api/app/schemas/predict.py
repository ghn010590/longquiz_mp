# app/schemas/predict.py
from typing import Any, List, Optional
from pydantic import BaseModel

class DataInputSchema(BaseModel):
    # Assume you have the following fields as an example
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'gender': "Female",
                        'age': 80,
                        'hypertension': 0,
                        'heart_disease': 1,
                        'smoking_history': "never",
                        'bmi': 25.19,
                        'HbA1c_level': 6.6,
                        'blood_glucose_level': 140
                    }
                ]
            }
        }
