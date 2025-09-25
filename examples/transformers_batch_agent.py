"""Example usage of the Web Gym environment."""

import functools

import tiktoken
import torch
from rich import print as rprint
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv
from aigym.types import RolloutBatch


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    def load_pipeline(model_id: str) -> Pipeline:
        return pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id,
                use_safetensors=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_id),
        )

    def policy(model_pipeline: Pipeline, prompt: str) -> RolloutBatch:
        model_pipeline(
            prompt,
            generation_kwargs={
                "temperature": 1.25,
                "max_new_tokens": 1024,
                "top_k": 64,
                "top_p": 0.95,
                "stop_strings": ["<end_of_turn>"],
            },
        )

    model_pipeline = load_pipeline("google/gemma-3-4b-it")
    agent = Agent(
        policy=functools.partial(policy, model_pipeline),
        token_encoder=enc,
        url_boundaries=["https://en.wikipedia.org"],
        stream=False,
    )

    n_hops = 1
    n_tries_per_hop = 5
    n_tries = n_hops * n_tries_per_hop
    env = WikipediaGymEnv(n_hops=n_hops)
    observation, info = env.reset()

    succeeded = False
    for step in range(1, n_tries + 1):
        pprint.print_observation(observation)
        pprint.print_context(observation)
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

    rprint(f"Finished after {n_tries} tries")
    if succeeded:
        rprint("✅ Target found")
    else:
        rprint("❌ Target not found")
    env.close()


if __name__ == "__main__":
    main()
