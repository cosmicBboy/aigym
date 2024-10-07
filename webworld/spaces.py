"""Definition of spaces in WebWorld."""

import re
from functools import lru_cache

import gymnasium as gym
import httpx
from markdownify import markdownify as md

from webworld.types import WebPage


RANDOM_URL = "https://en.wikipedia.org/wiki/Special:Random"


@lru_cache
def chunk_web_page(
    content: str,
    lines_per_chunk: int,
    overlap: int,
) -> list[str]:
    """Chunk a web page into smaller chunks.
    
    The chunking method implemented below is newline chunking using a sliding
    window.
    """
    lines = content.split("\n")
    chunks = []
    for i in range(0, len(lines), lines_per_chunk - overlap):
        chunks.append("\n".join(lines[i:i + lines_per_chunk]))
    return chunks


class WebGraph(gym.Space[WebPage]):
    """The space of web pages."""

    def __init__(
        self,
        start_url: str | None = None,
        text_format: str = "markdown",
        lines_per_chunk: int = 50,
        overlap: int = 40,
    ):
        """Initialize the web page."""
        self.start_url = start_url
        self.text_format = text_format
        self.lines_per_chunk = lines_per_chunk
        self.overlap = overlap
        self.session = httpx.Client()

    def sample(self) -> WebPage:
        """Sample a web page."""
        # RANDOM_URL will always return a random page even if you pass
        # the seed parameter passed into env.reset()
        return self.visit_url(self.start_url or RANDOM_URL)

    def visit_url(self, url: str):
        response = self.session.get(url, follow_redirects=True)
        # TODO: figure out how to select the main content of the page (exclude)
        # sidebar navigational elements, etc. Perhaps this can be done with
        # a function/subclass that uses beautifulsoup to select the main
        # content of the page.
        if self.text_format == "markdown":
            content = re.sub(r"\n+", "\n\n", md(response.text))
        else:
            raise ValueError(f"Text format '{self.text_format}' is not supported")
        content_chunks = chunk_web_page(
            content,
            self.lines_per_chunk,
            self.overlap,
        )
        return WebPage(
            url=str(response.url),
            content_chunks=content_chunks,
        )


class Tokens(gym.Space):
    """The space of language model tokens."""

    def __init__(self, tokenizer):
        """Initialize the text."""
        self.tokenizer = tokenizer

    def sample(self):
        """Sample a text."""
        return None
