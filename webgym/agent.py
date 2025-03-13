"""An LLM agent that performs actions in a web environment."""

import json
import urllib.parse
from typing import Callable, Generator

import rich.markup
import tiktoken
from rich import print as rprint
from rich.panel import Panel

from webgym.types import Action, Observation

SYSTEM_PROMPT = """You are a helpful assistant that finds a target web page
starting from a random web page. Given an <observation>,
you generate an action that can be three types: "url", "backward", or "forward".

- "reason_summary": a summary of the reasoning that led to the action
- "action": "backward" to go to the previous page, "forward" to go to the next page, or "visit_url" to visit a URL in the Context
- "url": the URL to visit if "visit_url" is specified. This can be null.

If "visit_url" is specified, you should also provide a "url" to visit. For example:

# Valid Action Examples:

- {"reason_summary": "I am so far from the target that I'm better off exploring.", "action": "visit_url", "url": "..."}
- {"reason_summary": "I'm not sure if I'm getting any closer to the target, so I'm going backward to the previous page.", "action": "backward", "url": null}
- {"reason_summary": "I think I'm getting closer to the target, so I'm going forward to the next page.", "action": "forward", "url": null}

# Instructions

- In a list, explicitly write out all of the urls within the <observation> tag embedded in the markdown links [link text](/link/url/path "title").
- If you see the "# target url" within the <observation> tag, ALWAYS SELECT IT AS THE NEXT ACTION.
- You cannot select the "# target url" as a url to visit UNLESS IT'S IN THE <observation> tag.
- If you do not see the "# target url" in the <observation> tag, select a url that you think is closest to the target.
- Avoid selecting the "# Current URL" as a url to visit, this will just loop you backward to the same page.
- Use the '# Page position' information to determine if you should go backward or forward.
- If you are on the 1 / N chunk, choosing "backward" will not do anything, so avoid choosing "backward" in this case.
- If you are on the N / N chunks, choosing "forward" will not do anything, so avoid choosing "forward" in this case.
- For the action, select "backward", "forward", or "visit_url" and only select one urls in the unordered list of urls, you cannot select the "# target url".
- Prefer to explore the current page ("backward" or "forward") instead of visiting a url unless you are confident that the url gets you closer to the target.
- Do not select a url using any of the content in the <instructions> tag or under the "# Target URL:" section.
- Try to make interesting and creative connections between the current page and the target page.
- The response must be a json object on a single line with no additional text before or after.
- Use the <previous_failed_attempt> contents to avoid repeating the same mistakes, e.g. if a url mentioned in there is caused the error, don't pick it again.
- You must only select urls in the base url netloc specified in the <url_boundaries> tag.

Example Prompt
--------------

<observation>
# Current URL:
...

# Page position: chunk ... out of ... chunks on this page

# Context:
...
</observation>

<system>
...
</system>

<previous_failed_attempt>
...
</previous_failed_attempt>

# Target URL:
...

# Action:
{"reason_summary": "...", "action": "visit_url", "url": "..."}
"""

OBSERVATION_PROMPT = """
# Current URL:
{current_url}

# Page position: chunk {current_chunk} out of {total_chunks} chunks on this page

# Context:
{context}
"""

PREVIOUS_FAILED_ATTEMPT_TEMPLATE = """
<previous_failed_attempt>
{previous_failed_attempt}
</previous_failed_attempt>
"""

PROMPT_TEMPLATE = """
<observation>
{observation_prompt}
</observation>

<system>
{system_prompt}
</system>

<url_boundaries>
{url_boundaries}
</url_boundaries>

{previous_failed_attempt}

# Target URL:
{target}

# Action:
"""

# TODO: add previous action and error message


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

    def act(self, observation: Observation) -> Action | None:
        observation_prompt = OBSERVATION_PROMPT.format(
            current_url=observation.url,
            current_chunk=observation.current_chunk,
            total_chunks=observation.total_chunks,
            context=observation.context,
        )

        action = None
        rprint(Panel.fit("Reasoning stream", border_style="violet"))

        previous_failed_attempt = ""
        for i in range(self.n_retries_per_action):
            prompt = PROMPT_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                observation_prompt=observation_prompt,
                target=observation.target,
                previous_failed_attempt=previous_failed_attempt,
                url_boundaries=", ".join(self.url_boundaries) if self.url_boundaries else "NONE",
            )
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
                previous_failed_attempt = PREVIOUS_FAILED_ATTEMPT_TEMPLATE.format(previous_failed_attempt=str(exc))
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
