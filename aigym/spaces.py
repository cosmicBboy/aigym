"""Definition of spaces in AIGym."""

import functools
import random
import re
import urllib.parse
from functools import lru_cache
from typing import Literal

import gymnasium as gym
import httpx
import markdown
import numpy as np
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from aigym.exceptions import NoPathsFoundError
from aigym.types import PageContent, WebPage


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


@lru_cache
def chunk_by_pattern(url: str, content: str, pattern: str) -> list[PageContent]:
    """Chunk a list of strings by a pattern."""

    url_path = urllib.parse.urlparse(url).path
    main_header = url_path.split("/")[-1].strip()

    chunks = []
    header = None
    chunk = ""
    _content_chunks = re.split(pattern, content)
    if not re.match(pattern, _content_chunks[0]):
        _content_chunks = [main_header, *_content_chunks]

    for i, (header, chunk) in enumerate(zip(_content_chunks[::2], _content_chunks[1::2])):
        if i > 0:
            header = header.strip().split("\n")[0].strip().replace(" ", "_")
        else:
            header = None

        chunks.append(
            PageContent(
                header=header,
                content=chunk,
                is_title_content=i == 0,
            )
        )

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

    def random_hop(
        self,
        page: WebPage,
        avoid_urls: set[str] | None = None,
        chunk_pattern: str | None = None,
    ) -> WebPage:
        """
        Randomly hop to a section of the same page or a new page via a link.
        """
        source_soup = BeautifulSoup(markdown.markdown(page.content), "html.parser")
        wiki_a_tags = source_soup.find_all(
            "a",
            href=lambda x: x is not None and self.link_filter(x),
        )
        paths_in_source_url = [a.attrs["href"] for a in wiki_a_tags]

        if avoid_urls:
            paths_in_source_url = [path for path in paths_in_source_url if path not in avoid_urls]

        page_chunks = [x for x in page.page_chunks if x.url != page.url and x.url not in avoid_urls]
        sample_from = [*paths_in_source_url, *page_chunks]

        # Create probabilities that balance sampling between paths and chunks
        n_paths = len(paths_in_source_url)
        n_chunks = len(page_chunks)
        if n_paths > 0 and n_chunks > 0:
            path_prob = 0.5 / n_paths
            chunk_prob = 0.5 / n_chunks
            probs = [path_prob] * n_paths + [chunk_prob] * n_chunks
        elif n_paths > 0:
            probs = [1.0 / n_paths] * n_paths
        elif n_chunks > 0:
            probs = [1.0 / n_chunks] * n_chunks
        else:
            probs = None

        if len(sample_from) == 0:
            raise NoPathsFoundError(page.url)

        _choice = np.random.choice(sample_from, p=probs)
        if isinstance(_choice, WebPage):
            return _choice

        _url = urllib.parse.urljoin(page.url, _choice)
        return self.get_page(_url, chunk_pattern)

    @functools.lru_cache
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

    @functools.lru_cache
    def get_page(
        self,
        url: str,
        chunk_pattern: str | None,
    ) -> WebPage:
        response = self.session.get(url, follow_redirects=True)

        content = self.get_soup(url)
        if self.text_format == "markdown":
            content = re.sub(r"\n+", "\n", md(str(content)))
        else:
            raise ValueError(f"Text format '{self.text_format}' is not supported")

        if chunk_pattern is None:
            content_chunks = [PageContent(header=None, content=content)]
        else:
            content_chunks = chunk_by_pattern(url, content, chunk_pattern)
            # TODO: move this into wikipedia-specific class
            content_chunks = [
                x for x in content_chunks if x.header not in ["References", "Footnotes", "See_also", "External_links"]
            ]
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
                {"id": "contentSub"},
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
