"""Environment for navigating the web."""

import urllib.parse
from typing import Any

import gymnasium as gym
from rich import print as rprint

from aigym.spaces import Tokens, WebGraph, WikipediaGraph
from aigym.types import Action, InternalEnvState, Observation


class Env(gym.Env):
    """AIGym environment."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        web_graph: WebGraph,
        n_hops: int | None = None,
        tokenizer: Any | None = None,
        render_mode: str | None = None,
        lines_per_chunk: int | None = None,
        overlap: int | None = None,
    ):
        """Initialize the environment.

        Args:
            web_graph: The web graph to use for the environment.
            n_hops: The start url will be sampled n_hops away from the target
                page. For each hop, the search ensures that the page links
                back to the previous page.
            tokenizer: The tokenizer to use for the action space.
            render_mode: The mode to render the environment in.
            lines_per_chunk: The number of lines per page chunk to return.
            overlap: The number of lines of overlap between chunks.
        """
        # this is a gym.Env attribute
        self.render_mode = render_mode

        # aigym-specific attributes
        self.observation_space: WebGraph = web_graph
        self.action_space: Tokens = Tokens(tokenizer=tokenizer)
        self.n_hops = n_hops
        self.lines_per_chunk = lines_per_chunk
        self.overlap = overlap
        self.travel_path = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize the window that will display the environment and the clock
        # to ensure the environment is rendered at the correct framerate in
        # human mode
        self.window = None
        self.clock = None

        self._state = InternalEnvState()

    @property
    def travel_map(self) -> dict:
        map = {}
        for current_url, next_url in zip(self.travel_path[:-1], self.travel_path[1:]):
            map[current_url] = next_url
        return map

    def _initialize_target_url(self, start_url: str, n_hops: int) -> tuple[str, list[str]]:
        travel_path = [start_url]
        _url = start_url
        print(f"Initializing target url {n_hops} hops away from {start_url}")
        for i in range(1, n_hops + 1):
            next_url = self.observation_space.random_hop(
                _url, set(travel_path + [urllib.parse.urlparse(x).path for x in travel_path])
            )
            travel_path.append(next_url)
            _url = next_url
            print(f"Hop {i} to {next_url}")
        print(f"Target url: {_url}")
        return _url, travel_path

    def random_start(self):
        self.start_url = str(
            self.observation_space.session.get(self.observation_space.RANDOM_URL, follow_redirects=True).url
        )

    def _get_observation(self):
        current_web_page = self.observation_space.get_page(
            self.start_url,
            self.lines_per_chunk,
            self.overlap,
        )

        # set new internal state
        self._state.current_web_page = current_web_page
        self._state.current_chunk_index = 0  # consider making this random

        context = self._state.current_web_page.content_chunks[self._state.current_chunk_index]

        observation = Observation(
            url=self._state.current_web_page.url,
            context=context,
            target_url=self.target_url,
            next_url=self.travel_map[self._state.current_web_page.url],
            current_chunk=self._state.current_chunk_index + 1,
            total_chunks=len(self._state.current_web_page.content_chunks),
        )
        info = {"travel_path": self.travel_path}
        return observation, info

    def reset_manual(
        self,
        start_url: str,
        target_url: str,
        travel_path: list[str],
    ):
        self.start_url = start_url
        self.target_url = target_url
        self.travel_path = travel_path
        return self._get_observation()

    def reset(
        self,
        start_url: str | None = None,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:
        """Reset the environment."""
        if start_url is not None:
            self.start_url = start_url
        else:
            self.random_start()

        self.target_url, self.travel_path = self._initialize_target_url(self.start_url, self.n_hops)
        observation, info = self._get_observation()
        rprint(f"reset current page to: {observation.url}")
        return observation, info

    def _current_page_is_target(self):
        _current_url = urllib.parse.urlparse(self._state.current_web_page.url)
        _target_url = urllib.parse.urlparse(self.target_url)
        return _current_url.netloc == _target_url.netloc and (
            _current_url.path == _target_url.path or _current_url.path.lower() == _target_url.path.lower()
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """Take a step in the environment."""
        if action.action == "backward":
            self._state.current_chunk_index = max(0, self._state.current_chunk_index - 1)
        elif action.action == "forward":
            self._state.current_chunk_index = min(
                len(self._state.current_web_page.content_chunks) - 1,
                self._state.current_chunk_index + 1,
            )
        elif action.action == "visit_url":
            self._state.current_web_page = self.observation_space.get_page(
                action.url,
                self.lines_per_chunk,
                self.overlap,
            )
            self._state.current_chunk_index = 0
        else:
            raise ValueError(f"invalid action: {action}")

        context = self._state.current_web_page.content_chunks[self._state.current_chunk_index]
        observation = Observation(
            url=self._state.current_web_page.url,
            context=context,
            target_url=self.target_url,
            next_url=self.travel_map[self._state.current_web_page.url],
            current_chunk=self._state.current_chunk_index + 1,
            total_chunks=len(self._state.current_web_page.content_chunks),
        )
        terminated = self._current_page_is_target()
        # alternatively, this would be distance to the target, but that would
        # require a routine to do random walks on the web graph starting from
        # the target
        reward = 1 if terminated else 0
        truncated = False
        info = {}

        rprint(
            f"Next observation: {observation.url}, position {observation.current_chunk} / {observation.total_chunks}"
        )
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""

    def close(self):
        """Close the environment."""


class WikipediaGymEnv(Env):
    """Wikipedia Gym environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            WikipediaGraph(),
            *args,
            **kwargs,
        )
