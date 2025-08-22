"""An LLM agent that performs actions in a web environment."""

import json
import logging
import urllib.parse
from typing import Callable, Generator

import httpx
import rich.markup
import tiktoken
from pydantic import ValidationError
from rich import print as rprint
from rich.panel import Panel

import aigym.prompts as prompts
from aigym.types import Action, ActionBatch, Observation, ParseType, RolloutBatch

logger = logging.getLogger(__name__)


class InvalidActionError(Exception):
    """Exception raised when a valid action has not been generated."""


class Agent:
    def __init__(
        self,
        policy: Callable[[str], Generator[str, None, None] | RolloutBatch],
        token_encoder: tiktoken.Encoding,
        url_boundaries: list[str] | None = None,
        stream: bool = True,
    ):
        self.policy = policy
        self.token_encoder = token_encoder
        self.url_boundaries = url_boundaries
        self.session = httpx.Client()
        self.stream = stream

    def perceive(self, observation: Observation) -> str:
        prompt = prompts.PERCEPTION_PROMPT_TEMPLATE.format(
            system_prompt=prompts.PERCEPTION_SYSTEM_PROMPT,
            observation=observation.context,
            target_url=observation.target_url,
        )
        stream = self.policy(prompt=prompt)
        output = ""

        rprint(Panel.fit("Perception stream", border_style="violet"))
        for chunk in stream:
            rprint(rich.markup.escape(chunk), end="")
            output += chunk

        print("\n")
        rprint(Panel.fit(rich.markup.escape(output), title="Perception", border_style="purple"))

        output = output.split("</think>")[-1].strip()
        return output

    def act(self, observation: Observation) -> Action | ActionBatch | None:
        if self.stream:
            action = self.action_from_stream(observation)
        else:
            action = self.action_from_batch(observation)

        return action

    def action_from_batch(
        self,
        observation: Observation,
    ) -> ActionBatch:
        prompt = self.get_prompt(observation)
        batch = self.policy(prompt)
        actions = []
        for completion in batch.completions:
            try:
                action_dict, completion, reasoning_trace, parse_type = self.parse_completion(completion, observation)
            except (json.JSONDecodeError, InvalidActionError) as exc:
                rprint(Panel.fit(f"[red]{type(exc)} Error: {exc}[/red]", border_style="red"))
                action_dict = None

            if action_dict is None:
                action = Action(completion=completion, parse_type="invalid")
            else:
                try:
                    action = Action(
                        **action_dict,
                        completion=completion,
                        reasoning_trace=reasoning_trace,
                        parse_type=parse_type,
                    )
                except (ValidationError, TypeError) as exc:
                    rprint(Panel.fit(f"[red]{type(exc)} Error: {exc}[/red]", border_style="red"))
                    action = Action(completion=completion, parse_type="invalid")

            actions.append(action)
        return ActionBatch(
            **batch.model_dump(),
            actions=actions,
        )

    def action_from_stream(
        self,
        observation: Observation,
    ) -> Action:
        prompt = self.get_prompt(observation)
        prompt_token_length = len(self.token_encoder.encode(prompt))
        rprint(Panel.fit(f"Prompt token length: {prompt_token_length}", border_style="violet"))

        stream = self.policy(prompt=prompt)

        rprint(Panel.fit("Action stream", border_style="violet"))
        completion = ""
        for chunk in stream:
            rprint(rich.markup.escape(chunk), end="")
            completion += chunk

        print("\n")
        rprint(Panel.fit("End attempt", border_style="purple"))
        try:
            action_dict, completion, reasoning_trace = self.parse_completion(completion, observation)
        except (json.JSONDecodeError, InvalidActionError) as exc:
            rprint(Panel.fit(f"[red]{type(exc)} Error: {exc}[/red]", border_style="red"))
            action_dict = None

        if action_dict is None:
            return Action(completion=completion)

        return Action(
            completion,
            **action_dict,
            reasoning_trace=reasoning_trace,
        )

    def get_prompt(self, observation: Observation) -> str:
        return prompts.WIKIPEDEA_ACTION_TEMPLATE.format(
            observation=observation.context,
            current_url=observation.url,
            current_chunk=observation.current_chunk,
            total_chunks=observation.total_chunks,
            target_url=observation.target_url,
            url_boundaries=", ".join(self.url_boundaries) if self.url_boundaries else "NONE",
        )

    def parse_completion(self, completion: str, observation: Observation) -> tuple[dict, str, str, ParseType]:
        import re

        # TODO: do exact match parsing.
        exact_match = re.search(r"<think>(.*?)</think>\n+<answer>(.*?)</answer>", completion, re.DOTALL)
        if exact_match:
            reasoning_trace = exact_match.group(1).strip()
            answer = exact_match.group(2).strip()
            try:
                action = json.loads(answer)
            except json.JSONDecodeError as exc:
                logger.info("Could not generate a valid action")
                raise InvalidActionError(str(exc)) from exc
            return action, completion, reasoning_trace, "exact_match"

        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning_trace = think_match.group(1).strip() if think_match else ""

        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        try:
            action = json.loads(answer)
        except json.JSONDecodeError as exc:
            logger.info("Could not generate a valid action")
            raise InvalidActionError(str(exc)) from exc

        try:
            action = {k.lower(): v for k, v in action.items()}
        except Exception as exc:
            raise InvalidActionError(str(exc)) from exc

        if action.get("action") != "visit_url" and "url" not in action:
            action["url"] = None

        if action.get("action") == "visit_url" and ("url" not in action or action["url"] is None):
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
        if not isinstance(action["url"], str):
            raise InvalidActionError(f"url is not a string. action: {action}")

        if action["url"] and not action["url"].startswith("http"):
            action["url"] = urllib.parse.urljoin(f"{_url.scheme}://{_url.netloc}", action["url"])

        try:
            if action["url"] and self._url_not_in_context(action["url"], observation.context):
                raise InvalidActionError(f"url {action['url']} is not in the context. action: {action}")
        except Exception as exc:
            raise InvalidActionError(str(exc)) from exc

        action_json = json.dumps(action, indent=2)
        # create clean completion
        completion = f"<think>\n{reasoning_trace}\n</think>\n<answer>\n{action_json}\n</answer>"
        return action, completion, reasoning_trace, "parseable"

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
