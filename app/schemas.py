from pydantic import BaseModel


class OCRBox(BaseModel):
    text: str
    confidence: float
    box: list[list[int]]


class OCRResponse(BaseModel):
    results: list[OCRBox]
