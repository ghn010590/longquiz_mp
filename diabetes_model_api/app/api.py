import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from diabetes_model import __version__ as model_version  # Adjust the import based on actual model package
from diabetes_model.predict import make_prediction

from app import __version__  # Ensure this is correctly defined somewhere in your application
from app.schemas import Health, PredictionResults, MultipleDataInputs  # Direct import to avoid issues
from app.config import settings

api_router = APIRouter()

@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root GET endpoint to check the health and version of the API.
    """
    health = Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )
    return health.dict()

@api_router.post("/predict", response_model=PredictionResults, status_code=200)
async def predict(input_data: MultipleDataInputs) -> Any:
    """
    Endpoint to make predictions with the diabetes model.
    """
    # Convert the input data to DataFrame, ensuring NaNs are handled properly
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    # Make prediction using the loaded model
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # Raise HTTPException if there are errors in the prediction
    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
