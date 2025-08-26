"""Example usage of the Web Gym environment."""

import os
from typing import Generator

import dotenv
import tiktoken
from google import genai
from google.genai import types
from rich import print as rprint

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


def main():
    dotenv.load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    enc = tiktoken.get_encoding("cl100k_base")

    def policy(prompt: str) -> Generator[str, None, None]:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=5000,
                temperature=0.2,
            ),
        ):
            delta = chunk.candidates[0].content.parts[0].text
            if delta is None:
                yield ""
                break
            yield delta

    agent = Agent(
        policy=policy,
        token_encoder=enc,
        url_boundaries=["https://en.wikipedia.org"],
    )

    env = WikipediaGymEnv(n_hops=3)
    # observation, info = env.reset()
    observation, info = env.reset_manual(
        start_url="https://en.wikipedia.org/wiki/Chenggong_Reservoir",
        target_url="https://en.wikipedia.org/wiki/Traditional_Chinese_characters",
        travel_path=[
            "https://en.wikipedia.org/wiki/Chenggong_Reservoir",
            "https://en.wikipedia.org/wiki/Water_Resources_Agency",
            "https://en.wikipedia.org/wiki/Traditional_Chinese_characters",
        ],
    )

    for step in range(1, 101):
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action = agent.act(observation)
        if action is None:
            rprint(f"No action taken at step {step}")
            continue
        pprint.print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            rprint(f"Episode terminated or truncated at step {step}")
            break

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    main()
