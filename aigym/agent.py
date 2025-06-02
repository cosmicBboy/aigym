"""An LLM agent that performs actions in a web environment."""

import json
import urllib.parse
from functools import partial
from typing import Callable, Generator

import httpx
import rich.markup
import tiktoken
from rich import print as rprint
from rich.panel import Panel

import aigym.prompts as prompts
from aigym.types import Action, Observation


class InvalidActionError(Exception):
    """Exception raised when a valid action has not been generated."""


class Agent:
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
        self.session = httpx.Client()

    def perceive(self, observation: Observation) -> str:
        prompt = prompts.PERCEPTION_PROMPT_TEMPLATE.format(
            system_prompt=prompts.PERCEPTION_SYSTEM_PROMPT,
            observation=observation.context,
            target_url=observation.target_url,
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
            prompts.WIKIPEDEA_ACTION_TEMPLATE.format,
            observation=observation.context,
            current_url=observation.url,
            current_chunk=observation.current_chunk,
            total_chunks=observation.total_chunks,
            target_url=observation.target_url,
            url_boundaries=", ".join(self.url_boundaries) if self.url_boundaries else "NONE",
        )

        action = None

        previous_failed_attempt: str = "None"
        for i in range(self.n_retries_per_action):
            prompt = _prompt_template(previous_failed_attempt=previous_failed_attempt)
            prompt_token_length = len(self.token_encoder.encode(prompt))
            rprint(Panel.fit(f"Prompt token length: {prompt_token_length}", border_style="violet"))
            rprint(Panel.fit(f"Attempt #{i + 1} / {self.n_retries_per_action}", border_style="purple"))

            stream = self.generate_function(
                prompt=prompt,
                next_url=observation.next_url,
                target_url=observation.target_url,
            )
            output = ""

            rprint(Panel.fit("Action stream", border_style="violet"))
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
        if "</think>" in response:
            reasoning_trace, _response = response.split("</think>", maxsplit=1)
            _response = _response.strip().replace("<think>", "").strip()
            _response = _response.replace("</think>", "").strip().replace("<tool_call>", "").strip()
        else:
            reasoning_trace = ""
            _response = response.strip()

        if _response.startswith("<answer>"):
            _response = _response.replace("<answer>", "").strip()
            _response = _response.replace("</answer>", "").strip()

        if _response.startswith(("```json")):
            _response = _response.replace("```json", "").replace("```", "").replace("json\n", "").strip()

        if _response.startswith(("```xml")):
            _response = _response.replace("```xml", "").replace("```", "").replace("xml\n", "").strip()

        if "```\n" in _response:
            _response = _response.replace("```\n", "").strip()

        try:
            action = json.loads(_response)
            action = {k.lower(): v for k, v in action.items()}

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

            if action["url"] and self._url_not_in_context(action["url"], observation.context):
                raise InvalidActionError(f"url {action['url']} is not in the context. action: {action}")

            return Action(**action, reasoning_trace=reasoning_trace)
        except json.JSONDecodeError as exc:
            raise InvalidActionError("Could not generate a valid action") from exc

    def _url_not_in_context(self, url: str, context: str) -> bool:
        _url = urllib.parse.urlparse(url)
        resolved_url = self.session.get(url, follow_redirects=True)
        _resolved_url = urllib.parse.urlparse(str(resolved_url.url))
        return (
            _url.path not in context
            and _url.path.lower() not in context.lower()
            and _resolved_url.path not in context
            and _resolved_url.path.lower() not in context.lower()
        )
