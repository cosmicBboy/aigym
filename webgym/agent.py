"""An LLM agent that performs actions in a web environment."""

import json
import urllib.parse
from functools import partial

import rich.markup
import tiktoken
from rich import print as rprint
from rich.panel import Panel

from webgym.types import Action, Observation


SYSTEM_PROMPT = """You are a helpful assistant that finds a target web page
starting from a random web page. Given an OBSERVATION (a Context and Target),
you generate an action that can be three types: "url", "back", or "forward".

- "reason_summary": a summary of the reasoning that led to the action
- "action": "back" to go to the previous page, "forward" to go to the next page, or "visit_url" to visit a URL in the Context
- "url": the URL to visit if "visit_url" is specified. This can be null.

If "visit_url" is specified, you should also provide a "url" to visit. For example:

# Valid Action Examples:

- {"reason_summary": "I am so far from the target that I'm better off exploring.", "action": "visit_url", "url": "..."}
- {"reason_summary": "I'm not sure if I'm getting any closer to the target, so I'm going back to the previous page.", "action": "back", "url": null}
- {"reason_summary": "I think I'm getting closer to the target, so I'm going forward to the next page.", "action": "forward", "url": null}


# Instructions

Use the '# Page position' information to determine if you should go back or forward.
If you are on the 1st chunk, choosing "back" will not do anything, so avoid choosing "back" in this case.
If you are on the Nth out of N chunks, choosing "forward" will not do anything, so avoid choosing "forward" in this case.
IN AN UNORDERED LIST, EXPLICITLY WRITE OUT ALL OF THE URLS WITHIN THE <observation> TAG EMBEDDED IN THE MARKDOWN LINKS [link text](url).
IF YOU SEE THE # Target URL WITHIN THE <observation> TAG, ALWAYS SELECT IT AS THE NEXT ACTION.
FOR THE ACTION, SELECT "back", "forward", OR "visit_url" AND ONLY SELECT ONE URLS IN THE UNORDERED LIST OF URLS, YOU CANNOT SELECT THE # Target URL.
PREFER TO EXPLORE THE CURRENT PAGE ("back" or "forward") INSTEAD OF VISITING A URL UNLESS YOU ARE CONFIDENT THAT THE URL GETS YOU CLOSER TO THE TARGET..
DO NOT SELECT A URL USING ANY OF THE CONTENT IN THE <instructions> TAG OR UNDER THE "# Target URL:" SECTION.
YOU CANNOT USE THE "# Target URL" TO DETERMINE WHAT TO DO NEXT.
TRY TO MAKE INTERESTING AND CREATIVE CONNECTIONS BETWEEN THE CURRENT PAGE AND THE TARGET PAGE.
The response MUST BE a JSON object on a single line with no additional text before or after.
USE THE <previous_failed_attempt> CONTENTS TO AVOID REPEATING THE SAME MISTAKES.
YOU MUST ONLY SELECT URLS IN THE BASE URL NETLOC SPECIFIC IN THE <url_boundaries> TAG.

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
        model_name: str,
        token_encoder: tiktoken.Encoding,
        n_retries_per_action: int = 30,
        url_boundaries: list[str] | None = None,
    ):
        self.model_name = model_name
        self.token_encoder = token_encoder
        self.n_retries_per_action = n_retries_per_action
        self.url_boundaries = url_boundaries
        self.init_model()

    def init_model(self):
        import ollama

        self.model = partial(ollama.generate, model=self.model_name, stream=True)

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
            rprint(Panel.fit(f"Attempt #{i+1} / {self.n_retries_per_action}", border_style="purple"))
            
            stream = self.model(prompt=prompt)
            output = ""

            for chunk in stream:
                rprint(rich.markup.escape(chunk.response), end="")
                output += chunk.response

            print("\n")
            rprint(Panel.fit(f"End attempt", border_style="purple"))
            try:
                action = self._parse_response(output, observation)
                break
            except (json.JSONDecodeError, InvalidActionError) as exc:
                rprint(Panel.fit(f"[red]{type(exc)} Error: {exc}[/red]", border_style="red"))
                previous_failed_attempt = PREVIOUS_FAILED_ATTEMPT_TEMPLATE.format(
                    previous_failed_attempt=str(exc)
                )
                continue

        if action is None:
            raise InvalidActionError("could not generate a valid action")
        return action

    def _parse_response(self, response: str, observation: Observation) -> Action:
        reasoning_trace, _response = response.split("</think>")
        _response = _response.strip().replace("<think>", "")

        for line in _response.split("\n"):
            try:
                action = json.loads(line)

                if action.get("action") != "visit_url" and "url" not in action:
                    action["url"] = None
                
                if action.get("action") == "visit_url" and action["url"] is None:
                    raise InvalidActionError(
                        f"url is required for visit_url action, found None. "
                        f"action: {action}"
                    )
                
                _url = urllib.parse.urlparse(observation.url)

                if self.url_boundaries is not None:
                    _url_boundary_netlocs = frozenset(
                        [
                            urllib.parse.urlparse(url_boundary).netloc
                            for url_boundary in self.url_boundaries
                        ]
                    )
                    if _url.netloc not in _url_boundary_netlocs:
                        raise InvalidActionError(
                            f"url {action['url']} is not in the url boundaries {self.url_boundaries}. "
                            f"action: {action}"
                        )

                # make sure url is a valid url
                if action["url"] and not action["url"].startswith("http"):
                    action["url"] = urllib.parse.urljoin(
                        f"{_url.scheme}://{_url.netloc}", action["url"]
                    )

                if (
                    action["url"]
                    and action["url"] not in observation.context
                    and urllib.parse.urlparse(action["url"]).path not in observation.context
                ):
                    raise InvalidActionError(
                        f"url {action['url']} is not in the context. "
                        f"action: {action}"
                    )

                return Action(**action, reasoning_trace=reasoning_trace)
            except json.JSONDecodeError:
                continue

        raise InvalidActionError("Could not generate a valid action")
