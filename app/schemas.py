from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    disease: str
    confidence: float
    recommendations: str
    next_steps: str
    tips: str
    disclaimer: str = (
        "This is not medical advice. "
        "Please consult a qualified dermatologist."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
