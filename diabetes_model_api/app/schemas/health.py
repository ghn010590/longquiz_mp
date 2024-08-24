# app/schemas/health.py
from pydantic import BaseModel

class Health(BaseModel):
    status: str = "available"
    api_version: str = "1.0"
