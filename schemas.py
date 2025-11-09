"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Job(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    mode: str = Field(..., description="processing mode: full or clips")
    target_duration_sec: int = Field(900, description="Desired duration for final edit in seconds")
    output_format: str = Field("16:9", description="Output aspect format")
    status: str = Field("queued", description="queued|processing|done|error")
    progress: float = Field(0.0, description="0..1 progress")
    recognized_game: Optional[str] = Field(None, description="Detected game title")
    source_files: List[str] = Field(default_factory=list, description="Uploaded source file paths")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="List of outputs: {video_url, cover_url, meta}")
    error: Optional[str] = None

class Asset(BaseModel):
    job_id: str = Field(...)
    kind: str = Field(..., description="video|cover")
    url: str = Field(...)
    meta: Dict[str, Any] = Field(default_factory=dict)
