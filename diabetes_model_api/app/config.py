import sys
from typing import List
from pydantic import AnyHttpUrl, BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    
    
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://localhost:8000"
    ]
    
    PROJECT_NAME: str = "Diabetes Prediction API"
    
    class Config:
        case_sensistive = True
        

settings = Settings()