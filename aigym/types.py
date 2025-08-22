"""Types for the AIGym environment."""

import urllib.parse
from typing import Literal

import torch
from pydantic import BaseModel, field_validator

ParseType = Literal["exact_match", "parseable", "invalid"]


class RolloutBatch(BaseModel, arbitrary_types_allowed=True):
    """Metadata for an action, used for training."""

    sequence_ids: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
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


class PageContent(BaseModel):
    header: str | None
    content: str
    is_title_content: bool = False


def _format_context(url: str, content: str, chunk_urls: list[str]) -> str:
    urls = "\n".join([f"- {url}" for url in chunk_urls])
    return f"""
Table of contents:
{urls}

Current contents: {url}
{content}
    """


class WebPage(BaseModel):
    """A web page."""

    url: str
    content_chunks: list[PageContent]
    content_chunk_index: int | None = None

    @property
    def context(self) -> str:
        if self.content_chunk_index is None:
            return f"Current contents:\n{self.content}"
        _context = self.content_chunks[self.content_chunk_index]
        chunk_urls = [x.url for x in self.page_chunks if x.url != self.url]
        return _format_context(self.url, _context.content, chunk_urls)

    @property
    def url_path(self) -> str:
        _url = urllib.parse.urlparse(self.url)
        return f"{_url.scheme}://{_url.netloc}/{_url.path}"

    @property
    def content_chunk_map(self) -> dict[str, PageContent]:
        return {x.header: x for x in self.content_chunks if x.header is not None}

    @property
    def content_chunk_header(self) -> str | None:
        if self.content_chunk_index is None:
            return None
        return self.content_chunks[self.content_chunk_index].header

    @property
    def content(self) -> str:
        if self.content_chunk_index is None:
            return "\n".join(x.content for x in self.content_chunks)
        return self.content_chunks[self.content_chunk_index].content

    @property
    def page_chunks(self) -> list["WebPage"]:
        """Get the paginated urls of the web page."""
        pages = []
        for i, chunk in enumerate(self.content_chunks):
            _parsed_url = urllib.parse.urlparse(self.url)
            path = f"{_parsed_url.path}#{chunk.header}" if chunk.header is not None else _parsed_url.path
            _url = urllib.parse.urljoin(f"{_parsed_url.scheme}://{_parsed_url.netloc}", path)
            pages.append(WebPage(url=_url, content_chunks=self.content_chunks, content_chunk_index=i))
        return pages


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
