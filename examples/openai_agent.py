"""Example usage of the Web Gym environment."""

from typing import Generator

import tiktoken
from openai import OpenAI
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    client = OpenAI()

    enc = tiktoken.get_encoding("cl100k_base")

    def generate_function(prompt: str) -> Generator[str, None, None]:
        for chunk in client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            max_tokens=2000,
            temperature=0.2,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta is None:
                yield ""
                break
            yield delta

    agent = Agent(
        generate_function=generate_function,
        token_encoder=enc,
        n_retries_per_action=10,
        url_boundaries=["https://en.wikipedia.org"],
    )

    env = WikipediaGymEnv()
    observation, info = env.reset_manual(
        start_url="https://en.wikipedia.org/wiki/Mammal",
        target_url="https://en.wikipedia.org/wiki/Dog",
        travel_path=["https://en.wikipedia.org/wiki/Mammal", "https://en.wikipedia.org/wiki/Dog"],
    )

    for step in range(1, 101):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            rprint(f"Episode terminated or truncated at step {step}")
            break

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    main()
