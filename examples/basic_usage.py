"""Example usage of the WebWorld environment."""

import gymnasium as gym
from webworld.env import WebWorldEnv
from webworld.agent import WebAgent


def main():
    env = WebWorldEnv(
        start_url="https://en.wikipedia.org/wiki/Main_Page",
        web_graph_kwargs={
            "lines_per_chunk": 50,
            "overlap": 10,
        }
    )
    agent = WebAgent("llama3.1", n_retries_per_action=100)

    observation, info = env.reset(seed=42)
    print(f"reset to: {observation.url}")

    for _ in range(20):
        action = agent.act(observation)
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
