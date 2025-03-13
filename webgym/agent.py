"""An LLM agent that performs actions in a web environment."""

import json
import urllib.parse
from functools import partial
from typing import Callable, Generator

import rich.markup
import tiktoken
from rich import print as rprint
from rich.panel import Panel

import webgym.prompts as prompts
from webgym.types import Action, Observation


class InvalidActionError(Exception):
    """Exception raised when a valid action has not been generated."""


class WebAgent:
    def __init__(
        self,
        generate_function: Callable[[str], Generator[str, None, None]],
        token_encoder: tiktoken.Encoding,
        n_retries_per_action: int = 30,
        url_boundaries: list[str] | None = None,
    ):
        self.generate_function = generate_function
        self.token_encoder = token_encoder
        self.n_retries_per_action = n_retries_per_action
        self.url_boundaries = url_boundaries

    def perceive(self, observation: Observation) -> str:
        prompt = prompts.PERCEPTION_PROMPT_TEMPLATE.format(
            system_prompt=prompts.PERCEPTION_SYSTEM_PROMPT,
            observation=observation.context,
            target=observation.target,
        )
        stream = self.generate_function(prompt=prompt)
        output = ""

        rprint(Panel.fit("Perception stream", border_style="violet"))
        for chunk in stream:
            rprint(rich.markup.escape(chunk), end="")
            output += chunk

        print("\n")
        rprint(Panel.fit(rich.markup.escape(output), title="Perception", border_style="purple"))

        output = output.split("</think>")[-1].strip()
        return output

    def act(self, observation: Observation) -> Action | None:
        _prompt_template = partial(
            prompts.ACTION_PROMPT_TEMPLATE.format,
            system_prompt=prompts.ACTION_SYSTEM_PROMPT,
            perception=self.perceive(observation),
            current_url=observation.url,
            current_chunk=observation.current_chunk,
            total_chunks=observation.total_chunks,
            target=observation.target,
            url_boundaries=", ".join(self.url_boundaries) if self.url_boundaries else "NONE",
        )

        action = None
        rprint(Panel.fit("Action stream", border_style="violet"))

        previous_failed_attempt: str = "None"
        for i in range(self.n_retries_per_action):
            prompt = _prompt_template(previous_failed_attempt=previous_failed_attempt)
            prompt_token_length = len(self.token_encoder.encode(prompt))
            rprint(Panel.fit(f"Prompt token length: {prompt_token_length}", border_style="violet"))
            rprint(Panel.fit(f"Attempt #{i + 1} / {self.n_retries_per_action}", border_style="purple"))

            stream = self.generate_function(prompt=prompt)
            output = ""

            for chunk in stream:
                rprint(rich.markup.escape(chunk), end="")
                output += chunk

            print("\n")
            rprint(Panel.fit("End attempt", border_style="purple"))
            try:
                action = self._parse_response(output, observation)
                break
            except (json.JSONDecodeError, InvalidActionError) as exc:
                rprint(Panel.fit(f"[red]{type(exc)} Error: {exc}[/red]", border_style="red"))
                previous_failed_attempt = str(exc)
                continue

        if action is None:
            raise InvalidActionError("could not generate a valid action")
        return action

    def _parse_response(self, response: str, observation: Observation) -> Action:
        reasoning_trace, _response = response.split("</think>")
        _response = _response.strip().replace("<think>", "").strip()

        if _response.startswith("```json"):
            _response = _response.replace("```json", "").replace("```", "").strip()

        try:
            action = json.loads(_response)

            if action.get("action") != "visit_url" and "url" not in action:
                action["url"] = None

            if action.get("action") == "visit_url" and action["url"] is None:
                raise InvalidActionError(f"url is required for visit_url action, found None. action: {action}")

            _url = urllib.parse.urlparse(observation.url)

            if self.url_boundaries is not None:
                _url_boundary_netlocs = frozenset(
                    [urllib.parse.urlparse(url_boundary).netloc for url_boundary in self.url_boundaries]
                )
                if _url.netloc not in _url_boundary_netlocs:
                    raise InvalidActionError(
                        f"url {action['url']} is not in the url boundaries {self.url_boundaries}. action: {action}"
                    )

            # make sure url is a valid url
            if action["url"] and not action["url"].startswith("http"):
                action["url"] = urllib.parse.urljoin(f"{_url.scheme}://{_url.netloc}", action["url"])

            if (
                action["url"]
                and action["url"] not in observation.context
                and urllib.parse.urlparse(action["url"]).path not in observation.context
            ):
                raise InvalidActionError(f"url {action['url']} is not in the context. action: {action}")

            return Action(**action, reasoning_trace=reasoning_trace)
        except json.JSONDecodeError as exc:
            raise InvalidActionError("Could not generate a valid action") from exc
