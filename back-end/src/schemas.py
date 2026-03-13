from pydantic import BaseModel, Field
from typing import List, Optional

#pydantic pra padronizar os textos recebidos
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto a ser classificado")

class PredictionResponse(BaseModel):
    predicted_class: int
    probabilities: List[float]
    class_names: Optional[List[str]] = None