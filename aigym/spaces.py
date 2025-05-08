"""Definition of spaces in AIGym."""

import random
import re
import urllib.parse
from functools import lru_cache
from typing import Literal

import gymnasium as gym
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from aigym.exceptions import NoPathsFoundError
from aigym.types import WebPage


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
        chunks.append("\n".join(lines[i : i + lines_per_chunk]))
    return chunks


class WebGraph(gym.Space[WebPage]):
    """The space of web pages."""

    def __init__(
        self,
        text_format: Literal["markdown", "html", "soup"] = "markdown",
        content_id: str | None = None,
        select_tags: list[str] | None = None,
        remove_attrs: list[dict[str, str]] | None = None,
        random_seed: int | None = None,
    ):
        """Initialize the web page.

        Args:
            url: The URL of the web page to visit.
            text_format: The format of the text to return.
            content_id: The ID of the main content to select from the web page.
            select_tags: The tags to select from the web page.
            remove_attrs: The attributes to remove from the web page.
            link_starts_with: The prefix of the link to select.
            link_ends_with: The suffix of the link to select.
            lines_per_chunk: The number of lines per chunk to return.
            overlap: The overlap between chunks.
            random_seed: The random seed to use.
        """
        self.text_format = text_format
        self.content_id = content_id
        self.select_tags = select_tags
        self.remove_attrs = remove_attrs
        self.random_seed = random_seed

        # create httpx session
        self.session = httpx.Client()

        # set random seed
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def link_filter(self, x: str) -> bool:
        raise NotImplementedError("link_filter must be implemented by the subclass")

    def random_hop(self, url: str, avoid_urls: set[str] | None = None):
        """
        Randomly hop to a new page.
        """
        source_soup = self.get_soup(url)

        wiki_a_tags = source_soup.find_all(
            "a",
            href=lambda x: x is not None and self.link_filter(x),
        )
        paths_in_source_url = [a.attrs["href"] for a in wiki_a_tags]
        if avoid_urls:
            paths_in_source_url = [path for path in paths_in_source_url if path not in avoid_urls]

        if len(paths_in_source_url) == 0:
            raise NoPathsFoundError(url)

        return urllib.parse.urljoin(url, random.choice(paths_in_source_url))

    def get_soup(self, url: str):
        response = self.session.get(url, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find(id=self.content_id)
        if content is None:
            raise ValueError(f"Content with id {self.content_id} not found")

        if self.remove_attrs:
            for attrs in self.remove_attrs:
                for tag in content.find_all(attrs=attrs):
                    tag.decompose()
        if self.select_tags:
            content.find_all(self.select_tags)
            content = BeautifulSoup("".join([str(c) for c in content]), "html.parser")
        return content

    def get_page(
        self,
        url: str,
        lines_per_chunk: int | None = None,
        overlap: int | None = None,
    ):
        response = self.session.get(url, follow_redirects=True)

        content = self.get_soup(url)
        if self.text_format == "markdown":
            content = re.sub(r"\n+", "\n", md(str(content)))
        else:
            raise ValueError(f"Text format '{self.text_format}' is not supported")

        if lines_per_chunk is None:
            content_chunks = [content]
        else:
            content_chunks = chunk_web_page(
                content,
                lines_per_chunk,
                overlap,
            )
        return WebPage(
            url=str(response.url),
            content_chunks=content_chunks,
        )


class WikipediaGraph(WebGraph):
    """The space of Wikipedia web pages."""

    RANDOM_URL = "https://en.wikipedia.org/wiki/Special:Random"

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            content_id="bodyContent",
            # only select main content
            select_tags=["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "table"],
            remove_attrs=[
                # remove any metadata elements, including infoboxes,
                # navboxes, and catlinks
                {"class": "navbox"},
                {"class": "catlinks"},
                {"class": "metadata"},
            ],
            **kwargs,
        )

    def link_filter(self, x: str) -> bool:
        return (
            x.startswith("/wiki/")
            and not x.endswith((".jpg", ".png", ".gif", ".svg"))
            and not x.startswith("/wiki/Wikipedia:")
            and not x.startswith("/wiki/Help:")
            and not x.startswith("/wiki/File:")
            and not x.startswith("/wiki/Category:")
            and not x.startswith("/wiki/Template:")
            and not x.startswith("/wiki/Portal:")
            and not x.startswith("/wiki/Special:")
            and not x.startswith("/wiki/Talk:")
            and not x.startswith("/wiki/User:")
        )


class Tokens(gym.Space):
    """The space of language model tokens."""

    def __init__(self, tokenizer):
        """Initialize the text."""
        self.tokenizer = tokenizer

    def sample(self):
        """Sample a text."""
        return None
