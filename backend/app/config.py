from pydantic import BaseModel

class Settings(BaseModel):
    sample_rate: int = 24000
    window_sec: float = 10.0
    stride_sec: float = 5.0
    max_windows: int = 5

    inconclusive_threshold: float = 0.55  
    binary_labels: tuple[str, str] = ("human", "pure_ai")

settings = Settings()
