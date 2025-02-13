"""Types for the WebGym environment."""

from typing import Literal

from pydantic import BaseModel, field_validator


class Action(BaseModel):
    """An action taken by the agent."""

    reason_summary: str
    action: Literal["visit_url", "back", "forward"]
    url: str | None = None
    reasoning_trace: str

    @field_validator("url")
    def validate_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("http"):
            raise ValueError("url must start with http")
        return v


class WebPage(BaseModel):
    """A web page."""

    url: str
    content_chunks: list[str]


class Observation(BaseModel):
    """The observation of the environment."""

    url: str
    context: str
    target: str
    current_chunk: int
    total_chunks: int


class InternalEnvState(BaseModel):
    """The internal state of the environment."""

    current_web_page: WebPage | None = None
    current_chunk_index: int | None = None
