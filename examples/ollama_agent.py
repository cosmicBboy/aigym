"""Example usage of the Web Gym environment."""

from typing import Generator

import ollama
import tiktoken
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
            stream=True,
            options={
                "temperature": 1.25,
                "top_k": 64,
                "top_p": 0.95,
            },
        ):
            yield chunk.response

    agent = Agent(
        policy=policy,
        token_encoder=enc,
        url_boundaries=["https://en.wikipedia.org"],
        stream=True,
    )

    n_hops = 2
    n_tries_per_hop = 5
    n_tries = n_hops * n_tries_per_hop
    env = WikipediaGymEnv(n_hops=n_hops)
    # observation, info = env.reset()
    # observation, info = env.reset_manual([
    #     "https://en.wikipedia.org/wiki/Macroglossum_wolframmeyi",
    #     "https://en.wikipedia.org/wiki/Taxonomy_(biology)",
    #     "https://en.wikipedia.org/wiki/Cladistics",
    # ])
    # observation, info = env.reset_manual([
    #     "https://en.wikipedia.org/wiki/Olive_A._Greeley",
    #     "https://en.wikipedia.org/wiki/Chile",
    #     "https://en.wikipedia.org/wiki/Chile#Geography",
    #     "https://en.wikipedia.org/wiki/Atacama_Desert",
    #     "https://en.wikipedia.org/wiki/Calama,_Chile",
    # ])
    observation, info = env.reset_manual(
        [
            "https://en.wikipedia.org/wiki/Atacama_Desert",
            "https://en.wikipedia.org/wiki/South_America",
            "https://en.wikipedia.org/wiki/Western_Hemisphere",
            # "https://en.wikipedia.org/wiki/Atacama_Desert#Republican_period",
            # "https://en.wikipedia.org/wiki/War_of_the_Pacific",
        ]
    )
    # TODO: handle invalid fragments, e.g. https://en.wikipedia.org/wiki/Atacama_Desert#Calama,
    # should fail and not actually visit the page.

    succeeded = False
    for step in range(1, n_tries + 1):
        pprint.print_observation(observation)
        # pprint.print_context(observation)
        action = agent.act(observation)
        if action.action is None:
            rprint(f"No action taken at step {step}")
            continue
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            succeeded = True
            rprint(f"Episode terminated or truncated at step {step}")
            break

    # TODO: hanld

    rprint(f"Finished after {n_tries} tries")
    if succeeded:
        rprint("✅ Target found")
    else:
        rprint("❌ Target not found")
    env.close()


if __name__ == "__main__":
    main()
