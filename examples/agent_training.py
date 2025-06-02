"""Example usage of the Web Gym environment."""

from functools import partial
from typing import Generator

import ollama
import tiktoken
import torch
from rich import print as rprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv


class GRPOTrainerRL(GRPOTrainer): ...


def train_agent(): ...


def reward_len(completions, **kwargs):
    # Define the reward function, which rewards completions that are close to 20 characters
    return [-abs(20 - len(completion)) for completion in completions]


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    print("Loading model")
    training_args = GRPOConfig(
        output_dir="gemma-3-4b-grpo",
        logging_steps=10,
        num_generations=4,
    )

    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_len],
        args=training_args,
    )

    def generate_function(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        next_url: str,
        target_url: str,
    ) -> Generator[str, None, None]:
        with torch.no_grad():
            ...
            # input schema:
            # {prompt: str, completion: str}
            # completion labels are not used
            # batch_inputs = [{"prompt": prompt, "next_url": next_url, "target_url": target_url}] * trainer.num_generations
            # TODO:
            # - Approach 1: compute loss based on per-step rewards
            # - Approach 2: compute loss based on target url found after n_tries

        # Under construction 🏗️
        for chunk in ollama.generate(
            # model="gemma3:27b",
            model="gemma3:12b",
            prompt=prompt,
            stream=True,
            options={
                "temperature": 2.0,
            },
        ):
            yield chunk.response

    agent = Agent(
        generate_function=partial(generate_function, trainer, model, trainer.tokenizer),
        token_encoder=enc,
        n_retries_per_action=20,
        url_boundaries=["https://en.wikipedia.org"],
    )

    n_hops = 1
    env = WikipediaGymEnv(n_hops=n_hops, lines_per_chunk=None)
    observation, info = env.reset()
    difficulty_factor = 5  # 1 is the hardest, higher numbers make it easier
    n_tries = int(n_hops * difficulty_factor)
    n_groups = 8

    print("Starting to train")
    group_rewards = []
    for _ in range(n_groups):
        rewards = []
        for step in range(1, n_tries):
            pprint.print_observation(observation)
            pprint.print_context(observation)
            action = agent.act(observation)
            pprint.print_action(action)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                rewards.append(reward)
                rprint(f"Episode terminated or truncated at step {step}")
                break

        # reward is defined as whether or not the agent found the target url
        # within the number of tries
        group_rewards.append(max(rewards))

        """
        # TODO:
        - collect n trajectories for each observation
        - rewards:
          - valid format: 0.25
          - valid action: 0.25
          - correct target url: 1.0
        - training:
          - compute advantages based on variable-length trajectories
          - broadcast advantages to all all the steps within a trajectory
          - compute GRPO loss for all groups, outputs, and steps
          - do gradient accumulation over each group

        # Order of operations
        - Generate completions for each step in the trajectory
        - Two approaches to reward:
          - Approach 1: Reward is based on whether the action contains the next
            url within the true trajectory.
          - Approach 2: Wait until n_tries, then reward is based on whether the agent
            found the target url regardless of whether the agent traversed the
            exact path.
        - Collect the rewards per group, advantages.
        - Compute GRPO loss for all groups, outputs, and steps
        - Treat each step in a trajectory as independent.

        # Notes
        - Modify the GRPOTrainer. The main methods of interest are:
          - GRPOTrainer._prepare_inputs
          - GRPOTrainer._generate_and_score_completions
          - GRPOTrainer.compute_loss
        - Take a look at Trainer:
          - Trainer.training_step is the main method that computes the loss on
            a batch of inputs. Understand how _prepare_inputs and compute_loss
            are used to compute the loss.
        """

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    main()
