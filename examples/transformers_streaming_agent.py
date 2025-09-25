"""Example usage of the Web Gym environment."""

import functools
from threading import Thread
from typing import Generator

import tiktoken
import torch
from rich import print as rprint
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, TextIteratorStreamer, pipeline

import aigym.pprint as pprint
import aigym.prompts as prompts
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


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

    def policy(pl: Pipeline, prompt: str) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(pl.tokenizer, skip_prompt=True)
        chat_messages = [
            {
                "role": "system",
                "content": prompts.REASONING_TEMPLATE,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        chat_prompt = pl.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        thread = Thread(
            target=pl,
            args=(chat_prompt,),
            kwargs={
                "generation_kwargs": {
                    "temperature": 1.25,
                    "max_new_tokens": 1024,
                    "top_k": 64,
                    "top_p": 0.95,
                    "stop_strings": ["<end_of_turn>"],
                },
                "streamer": streamer,
            },
        )
        thread.start()
        for chunk in streamer:
            yield chunk

    pl = load_pipeline("google/gemma-3-4b-it")
    agent = Agent(
        policy=functools.partial(policy, pl),
        token_encoder=enc,
        url_boundaries=["https://en.wikipedia.org"],
    )

    n_hops = 2
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

    rprint(f"Finished after {step} tries")
    if succeeded:
        rprint("✅ Target found")
    else:
        rprint("❌ Target not found")
    env.close()


if __name__ == "__main__":
    main()
