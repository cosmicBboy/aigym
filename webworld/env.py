"""Environment for navigating the web."""

from typing import Any

import gymnasium as gym
from webworld.spaces import Tokens, WebGraph
from webworld.types import Action, InternalEnvState, Observation, WebPage

DEFAULT_TARGET = "https://en.wikipedia.org/wiki/Dog"


class WebWorldEnv(gym.Env):
    """WebWorld environment."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        start_url: str | None = None,
        tokenizer: Any | None = None,
        render_mode: str | None = None,
        web_graph_kwargs: dict | None = None,
    ):
        """Initialize the environment."""
        self.render_mode = render_mode

        self.observation_space: WebGraph = WebGraph(start_url=start_url, **web_graph_kwargs)
        self.action_space: Tokens = Tokens(tokenizer=tokenizer)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize the window that will display the enviornment and the clock
        # to ensure the environment is rendered at the correct framerate in
        # human mode
        self.window = None
        self.clock = None
        
        self._state = InternalEnvState()

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment."""
        current_web_page: WebPage = self.observation_space.sample()

        # set new internal state
        self._state.current_web_page = current_web_page
        self._state.current_chunk_index = 0  # consider making this random

        context = self._state.current_web_page.content_chunks[
            self._state.current_chunk_index
        ]

        # TODO: reset target to another random page

        observation = Observation(
            url=self._state.current_web_page.url,
            context=context,
            target=DEFAULT_TARGET,
        )
        # TODO: implement info as the distance (definition TBD) to the target text
        info = {}
        # replace more than 2 newlines with a single newline
        return observation, info

    def step(self, action: Action):
        """Take a step in the environment."""
        if action.action == "back":
            self._state.current_chunk_index = max(
                0, self._state.current_chunk_index - 1
            )
        elif action.action == "forward":
            self._state.current_chunk_index = min(
                len(self._state.current_web_page.content_chunks) - 1,
                self._state.current_chunk_index + 1,
            )
        elif action.action == "visit_url":
            self._state.current_web_page = self.observation_space.visit_url(action.url)
            self._state.current_chunk_index = 0
        else:
            raise ValueError(f"invalid action: {action}")
        
        context = self._state.current_web_page.content_chunks[
            self._state.current_chunk_index
        ]
        observation = Observation(
            url=self._state.current_web_page.url,
            context=context,
            target=DEFAULT_TARGET,
        )
        terminated = self._state.current_web_page.url == DEFAULT_TARGET
        # alternatively, this would be distance to the target, but that would
        # require a routine to do random walks on the web graph starting from
        # the target
        reward = 0 if terminated else -1
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""

    def close(self):
        """Close the environment."""
