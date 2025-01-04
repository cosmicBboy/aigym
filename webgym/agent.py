"""An LLM agent that performs actions in a web environment."""

import json
import urllib.parse
from functools import partial

from webgym.types import Action, Observation


SYSTEM_PROMPT = """You are a helpful assistant that finds a target web page
starting from a random web page. Given an OBSERVATION (a Context and Target),
you generate an action that can be three types: "url", "back", or "forward".

- "visit_url": visit a URL in the Context
- "back": go back to the previous page
- "forward": go forward to the next page

If "visit_url" is specified, you should also provide a "url" to visit. For example:

# Valid Actions:
- {"reasoning": "I am so far from the target that I'm better off exploring.", "action": "visit_url", "url": "https://en.wikipedia.org/wiki/Special:Random"}
- {"reasoning": "I'm not sure if I'm getting any closer to the target, so I'm going back to the previous page.", "action": "back", "url": null}
- {"reasoning": "I think I'm getting closer to the target, so I'm going forward to the next page.", "action": "forward", "url": null}

# Instructions:

Only select urls embedded in the markdown links [link text](url).

MAKE SURE the selected url starts with "http://" or "https://". For links that
start with "/", i.e. don't start with "http://" or "https://", you should use
the base url based on the "# Current URL:".

DO NOT generate actions or urls using any of the content in the System Prompt or
under the "# Target URL:" section.

ONLY GENERATE ACTIONS as json objects in the form:
{"reasoning": str, "action": oneof("visit_url", "back", "forward"), "url": str | null}

An example OBSERVATION would be:

# Current URL:
https://en.wikipedia.org/wiki/Main_Page

# Context:
<context>

# Target URL:
<target>

# Action:
{"reasoning": "...", "action": "visit_url", "url": "https://en.wikipedia.org/wiki/Wikipedia:About"}

The response MUST BE a JSON object with no additional text before or after.
"""

# TODO: the reasoning needs to be tokens output before the action.

OBSERVATION_PROMPT = """
# Current URL:
{current_url}

# Context:
{context}

# Target URL:
{target}

# Action:
"""

PROMPT_TEMPLATE = """
<SYSTEM_PROMPT>
{system_prompt}
</SYSTEM_PROMPT>

<OBSERVATION>
{observation_prompt}
"""


class InvalidActionError(Exception):
    """Exception raised when a valid action has not been generated."""


class WebAgent:

    def __init__(self, model_name: str, n_retries_per_action: int = 30):
        self.model_name = model_name
        self.n_retries_per_action = n_retries_per_action
        self.init_model()

    def init_model(self):
        import ollama

        self.model = partial(ollama.generate, model=self.model_name)

    def act(self, observation: Observation) -> Action | None:
        # TODO: use a tokenizer to count tokens in an observation

        observation_prompt = OBSERVATION_PROMPT.format(
            current_url=observation.url,
            context=observation.context,
            target=observation.target,
        )

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            observation_prompt=observation_prompt,
        )

        action = None
        for _ in range(self.n_retries_per_action):
            output = self.model(prompt=prompt)
            try:
                action = self._parse_response(output["response"], observation)
                break
            except (json.JSONDecodeError, InvalidActionError):
                continue

        if action is None:
            raise InvalidActionError("could not generate a valid action")
        return action

    def _parse_response(self, response: str, observation: Observation) -> Action:
        for line in response.split("\n"):
            try:
                action = json.loads(line)
                assert "url" in action

                # make sure url is a valid url
                if action["url"] and not action["url"].startswith("http"):
                    _url = urllib.parse.urlparse(observation.url)
                    action["url"] = urllib.parse.urljoin(
                        f"{_url.scheme}://{_url.netloc}", action["url"]
                    )

                # make sure url is in the context
                if action["url"] and action["url"] not in observation.context:
                    raise InvalidActionError("url is not in the context")
                
                if action["action"] == "visit_url" and action["url"] is None:
                    raise InvalidActionError("url is required for visit_url action")

                return Action(**action)
            except json.JSONDecodeError:
                continue

        raise InvalidActionError("Could not generate a valid action")
