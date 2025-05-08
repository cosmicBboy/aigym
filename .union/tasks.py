"""Example usage of the Web Gym environment."""

from typing import Generator

import client_streaming
import tiktoken
import union
from rich import print as rprint
from rich.panel import Panel

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import Env


@union.task
def wiki_link_search(endpoint_url: str):
    # other start urls:
    # https://en.wikipedia.org/wiki/Mammal
    # https://en.wikipedia.org/wiki/Canidae
    # https://en.wikipedia.org/wiki/Vertebrate

    env = Env(
        start_url="https://en.wikipedia.org/wiki/Vertebrate",
        target_url="https://en.wikipedia.org/wiki/Dog",
        web_graph_kwargs={
            "lines_per_chunk": 100,
            "overlap": 0,
        },
    )

    enc = tiktoken.get_encoding("cl100k_base")

    def generate_function(prompt: str) -> Generator[str, None, None]:
        yield from client_streaming.run(url=endpoint_url, message=prompt)

    agent = Agent(
        generate_function=generate_function,
        token_encoder=enc,
        n_retries_per_action=10,
        url_boundaries=["https://en.wikipedia.org"],
    )

    observation, info = env.reset(seed=42)
    rprint(f"reset current page to: {observation.url}")

    for step in range(1, 101):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rprint(
            f"Next observation: {observation.url}, position {observation.current_chunk} / {observation.total_chunks}"
        )
        if terminated or truncated:
            rprint(Panel.fit(f"Episode terminated or truncated at step {step}", border_style="spring_green3"))
            break
            # observation, info = env.reset()

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    wiki_link_search(endpoint_url="https://shy-violet-c90de.apps.serverless-1.us-east-2.s.union.ai")
