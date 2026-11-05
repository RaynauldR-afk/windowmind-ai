from typing import TypedDict, NotRequired
from pydantic import BaseModel, Field

class WindowRecord(BaseModel):
    window_id: int
    start_char: int
    end_char: int
    text: str
    summary: str = ""

class WindowStats(BaseModel):
    total_characters: int
    total_windows: int
    window_size: int
    overlap: int
    stride: int
    average_window_chars: float
    processing_seconds: float

class AgentState(TypedDict):
    document_text: str
    question: str
    windows: NotRequired[list[dict]]
    summaries: NotRequired[list[str]]
    retrieved_context: NotRequired[str]
    answer: NotRequired[str]
    stats: NotRequired[dict]
    error: NotRequired[str]

class QueryRequest(BaseModel):
    document: str = Field(min_length=1)
    question: str = Field(min_length=1)
