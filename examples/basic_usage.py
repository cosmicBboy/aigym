"""Example usage of the Web Gym environment."""

import gymnasium as gym
from webgym.env import WebGymEnv
from webgym.agent import WebAgent


def main():
    env = WebGymEnv(
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
