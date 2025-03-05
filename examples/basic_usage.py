"""Example usage of the Web Gym environment."""

import tiktoken
import webgym.pprint as pprint
from webgym.agent import WebAgent
from webgym.env import WebGymEnv
from rich import print as rprint
from rich.panel import Panel


def main():
    env = WebGymEnv(
        # start_url="https://en.wikipedia.org/wiki/Mammal",
        # start_url="https://en.wikipedia.org/wiki/Canidae",
        start_url="https://en.wikipedia.org/wiki/Vertebrate",
        target_url="https://en.wikipedia.org/wiki/Dog",
        web_graph_kwargs={
            "lines_per_chunk": 100,
            "overlap": 0,
        }
    )

    enc = tiktoken.get_encoding("cl100k_base")
    agent = WebAgent(
        "deepseek-r1:14b",
        token_encoder=enc,
        n_retries_per_action=5,
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
            f"Next observation: {observation.url}, "
            f"position {observation.current_chunk} / {observation.total_chunks}"
        )
        if terminated or truncated:
            rprint(Panel.fit(f"Episode terminated or truncated at step {step}", border_style="spring_green3"))
            break
            # observation, info = env.reset()

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    main()
