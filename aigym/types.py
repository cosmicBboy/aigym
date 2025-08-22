"""Types for the AIGym environment."""

from typing import Literal

import torch
from pydantic import BaseModel, field_validator

ParseType = Literal["exact_match", "parseable", "invalid"]


class RolloutBatch(BaseModel, arbitrary_types_allowed=True):
    """Metadata for an action, used for training."""

    sequence_ids: torch.Tensor
    input_ids: torch.Tensor
    completions: list[str]
    log_probs: torch.Tensor | None = None


class Action(BaseModel):
    """An action taken by the agent."""

    completion: str
    reason_summary: str | None = None
    action: Literal["visit_url", "backward", "forward"] | None = None
    url: str | None = None
    reasoning_trace: str | None = None
    parse_type: ParseType | None = None

    @field_validator("url")
    def validate_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("http"):
            raise ValueError("url must start with http")
        return v


class ActionBatch(RolloutBatch):
    actions: list[Action]


class WebPage(BaseModel):
    """A web page."""

    url: str
    content_chunks: list[str]


class Observation(BaseModel):
    """The observation of the environment."""

    url: str
    context: str
    next_url: str | None
    target_url: str
    current_chunk: int
    total_chunks: int


class InternalEnvState(BaseModel):
    """The internal state of the environment."""

    current_web_page: WebPage | None = None
    current_chunk_index: int | None = None
