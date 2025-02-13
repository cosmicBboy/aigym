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

- {"reason_summary": "I am so far from the target that I'm better off exploring.", "action": "visit_url", "url": "https://en.wikipedia.org/wiki/Special:Random"}
- {"reason_summary": "I'm not sure if I'm getting any closer to the target, so I'm going back to the previous page.", "action": "back", "url": null}
- {"reason_summary": "I think I'm getting closer to the target, so I'm going forward to the next page.", "action": "forward", "url": null}

Use the '# Page position' information to determine if you should go back or forward.
If you are on the 1st chunk, choosing "back" will not do anything, so avoid choosing "back" in this case.
If you are on the Nth out of N chunks, choosing "forward" will not do anything, so avoid choosing "forward" in this case.

# Instructions:

Only select urls embedded in the markdown links [link text](url).

MAKE SURE the selected url starts with "http://" or "https://". For links that
start with "/", i.e. don't start with "http://" or "https://", you should use
the base url based on the "# Current URL:".

DO NOT generate actions or urls using any of the content in the System Prompt or under the "# Target URL:" section.
YOU CANNOT USE SEARCH FUNCTIONALITY TO DETERMINE WHAT TO DO NEXT.
YOU CANNOT USE THE "# Target URL" IN THE CONTEXT TO DETERMINE WHAT TO DO NEXT, ONLY LINKS IN THE "# Context".
FULLY EXPLORE A PAGE BEFORE CHOOSING A URL TO VISIT.
ALWAYS TRY TO MAKE INTERESTING AND CREATIVE CONNECTIONS BETWEEN THE CURRENT PAGE AND THE TARGET PAGE.
The response MUST BE a JSON object on a single line with no additional text before or after.

<EXAMPLE OBSERVATION>

# Current URL:
https://en.wikipedia.org/wiki/Main_Page

# Context:
<context>

# Target URL:
<target>

# Action:
{"reason_summary": "...", "action": "visit_url", "url": "https://en.wikipedia.org/wiki/Wikipedia:About"}
"""

OBSERVATION_PROMPT = """
# Current URL:
{current_url}

# Page position: chunk {current_chunk} out of {total_chunks} chunks on this page

# Context:
{context}
"""

PROMPT_TEMPLATE = """
<observation>
{observation_prompt}
</observation>

<instructions>
{system_prompt}
</instructions>

# Target URL:
{target}

# Action:
"""


class InvalidActionError(Exception):
    """Exception raised when a valid action has not been generated."""


class WebAgent:

    def __init__(self, model_name: str, token_encoder: tiktoken.Encoding, n_retries_per_action: int = 30):
        self.model_name = model_name
        self.token_encoder = token_encoder
        self.n_retries_per_action = n_retries_per_action
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

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            observation_prompt=observation_prompt,
            target=observation.target,
        )

        action = None
        prompt_token_length = len(self.token_encoder.encode(prompt))
        rprint(Panel.fit(f"Prompt token length: {prompt_token_length}", border_style="violet"))
        rprint(Panel.fit("Reasoning stream", border_style="violet"))

        for i in range(self.n_retries_per_action):
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

                if action["action"] != "visit_url" and "url" not in action:
                    action["url"] = None
                
                if action["action"] == "visit_url" and action["url"] is None:
                    raise InvalidActionError(f"url is required for visit_url action, found None")

                # make sure url is a valid url
                if action["url"] and not action["url"].startswith("http"):
                    _url = urllib.parse.urlparse(observation.url)
                    action["url"] = urllib.parse.urljoin(
                        f"{_url.scheme}://{_url.netloc}", action["url"]
                    )

                if (
                    action["url"]
                    and action["url"] not in observation.context
                    and urllib.parse.urlparse(action["url"]).path not in observation.context
                ):
                    raise InvalidActionError(f"url {action['url']} is not in the context")

                return Action(**action, reasoning_trace=reasoning_trace)
            except json.JSONDecodeError:
                continue

        raise InvalidActionError("Could not generate a valid action")
