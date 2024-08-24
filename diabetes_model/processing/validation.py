import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from diabetes_model.config.core import config

class DataInputSchema(BaseModel):
    gender: Optional[str]
    age: Optional[float]
    hypertension: Optional[int]
    heart_disease: Optional[int]
    smoking_history: Optional[str]
    bmi: Optional[float]
    HbA1c_level: Optional[float]
    blood_glucose_level: Optional[int]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]