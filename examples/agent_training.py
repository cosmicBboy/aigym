"""Example usage of the Web Gym environment.

GRPO implementation adapted from:
https://github.com/open-thought/tiny-grpo

Usage:

(Mac) Set MPS as the default device:
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Run the script:
```
python examples/agent_training.py
```
"""

from dataclasses import dataclass
from functools import partial
from typing import cast

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from rich import print as rprint
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, PreTrainedModel

import aigym.pprint as pprint
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv
from aigym.types import Action, ActionBatch, Observation, RolloutBatch


@dataclass
class TrainingArgs:
    optim: str = "adamw"
    lr: float = 1e-4
    group_size: int = 4
    advantage_eps: float = 1e-8
    clip_eps: float = 0.2
    kl_weight: float = 0.01
    max_grad_norm: float = 5.0
    use_bnb_quantization: bool = False
    enable_gradient_checkpointing: bool = False


def masked_mean(
    tensor: torch.Tensor,
    action_mask: torch.Tensor | None,
    dim: int = None,
) -> torch.Tensor:
    if action_mask is None:
        return tensor.mean(axis=dim)
    return (tensor * action_mask).sum(axis=dim) / action_mask.sum(axis=dim)


def copy_model(
    model: PreTrainedModel | PeftModelForCausalLM,
    base_model: PreTrainedModel = None,
    lora_config: LoraConfig = None,
) -> PreTrainedModel:
    if isinstance(model, PeftModelForCausalLM):
        model = cast(PeftModelForCausalLM, model)
        model_copy = get_peft_model(base_model, lora_config)
        model_copy.load_state_dict({k: v.clone() if "lora" in k else v for k, v in model.state_dict().items()})
    else:
        model_copy = type(model)(model.config)
        model_copy.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})

    return model_copy


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_ref: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - log_probs_old).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()


def reward_function(action: Action | None, observation: Observation) -> float:
    """Reward function.

    - no/invalid action = 0
    - correctly formatted = 0.25
    - correct url = 1.0
    - target url = 2.0
    """
    if action is None:
        return 0
    elif action.url == observation.next_url:
        return 1.0
    elif action.url == observation.target_url:
        return 2.0
    else:
        # correctly formatted, not next url and target url
        return 0.25


@torch.no_grad()
def policy(
    training_args: TrainingArgs,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    generation_config: GenerationConfig,
    prompt: str,
) -> RolloutBatch:
    # tokenize and prepare inputs for batch generation
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(training_args.group_size, 1)
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(training_args.group_size, 1)

    # generate completions
    model.eval()
    with torch.no_grad():
        sequence_ids = model.generate(**model_inputs, generation_config=generation_config)

    completions = tokenizer.batch_decode(
        sequence_ids[:, model_inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    # action mask makes sure end of sequence tokens are masked out of the
    # loss calculation
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == tokenizer.eos_token_id] = False
    action_mask = action_mask[:, 1:]

    return RolloutBatch(
        sequence_ids=sequence_ids,
        action_mask=action_mask,
        completions=completions,
    )


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Reference: http://joschu.net/blog/kl-approx.html
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1


def compute_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"][:, :-1].to(model.dtype)
    output_ids = sequence_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def update_policy(
    optimizer: optim.Optimizer,
    tokenizer: AutoTokenizer,
    objective: GRPOLoss,
    training_args: TrainingArgs,
    model: PreTrainedModel,
    previous_model: PreTrainedModel | None,
    reference_model: PreTrainedModel,
    action_batch: ActionBatch,
    rewards: list[float],
):
    model.train()
    optimizer.zero_grad()
    rewards = torch.tensor(rewards, dtype=model.dtype).unsqueeze(1)
    advantages = rewards - rewards.mean() / (rewards.std() + training_args.advantage_eps)

    pad_token_id = tokenizer.eos_token_id
    attention_mask = action_batch.sequence_ids != pad_token_id
    log_probs = compute_log_probs(model, action_batch.sequence_ids, attention_mask)
    with torch.no_grad():
        log_probs_old = (
            log_probs
            if previous_model is None
            else compute_log_probs(previous_model, action_batch.sequence_ids, attention_mask)
        )
        log_probs_ref = compute_log_probs(reference_model, action_batch.sequence_ids, attention_mask)

    loss, kl = objective(
        log_probs=log_probs,
        log_probs_old=log_probs_old,
        log_probs_ref=log_probs_ref,
        advantages=advantages.to(model.device),
        action_mask=action_batch.action_mask,
    )

    if not loss.isfinite():
        print(f"Loss not finite, skipping backward, loss={loss}, advantages: {advantages}")
        return previous_model

    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=training_args.max_grad_norm)
    print(f"kl={kl: .4f}, grad_norm={grad_norm: .4f}")
    optimizer.step()
    return model


def main(training_args: TrainingArgs):
    enc = tiktoken.get_encoding("cl100k_base")

    print("Loading model")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # model_id = "google/gemma-3-1b-it"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    target_lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_lora_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    bnb_config = None
    if training_args.use_bnb_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    reference_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        quantization_config=bnb_config,
    ).to(device)

    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        quantization_config=bnb_config,
    ).to(device)

    reference_model = get_peft_model(base_model, lora_config, adapter_name="default")
    model = get_peft_model(base_model, lora_config, adapter_name="default")
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    reference_model.eval()

    if training_args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if training_args.optim == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=training_args.lr)
    elif training_args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=training_args.lr)
    else:
        raise ValueError(f"Invalid optimizer: {training_args.optim}")

    objective = GRPOLoss(
        clip_eps=training_args.clip_eps,
        kl_weight=training_args.kl_weight,
    )

    generation_config = GenerationConfig(
        do_sample=True,
        min_new_tokens=64,
        max_new_tokens=128,
        temperature=1.25,
        padding=True,
        padding_size="left",
        return_attention_mask=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(f"Initializing agent with generation config {generation_config}")
    agent = Agent(
        policy=partial(
            policy,
            training_args,
            model,
            tokenizer,
            generation_config,
        ),
        stream=False,
        token_encoder=enc,
        url_boundaries=["https://en.wikipedia.org"],
    )

    n_hops = 1
    env = WikipediaGymEnv(n_hops=n_hops, lines_per_chunk=None)
    observation, info = env.reset()
    difficulty_factor = 5  # 1 is the hardest, higher numbers make it easier
    n_tries = int(n_hops * difficulty_factor)

    print(f"Starting to train with {n_tries} steps")
    previous_model = None
    for step in range(1, n_tries):
        print(f"step {step}")
        pprint.print_observation(observation)
        pprint.print_context(observation)
        action_batch: ActionBatch = agent.act(observation)

        step_action: Action | None = None
        rewards: list[float] = []
        for action in action_batch.actions:
            reward = reward_function(action, observation)
            rewards.append(reward)
            if action is None:
                continue
            pprint.print_action(action)
            if action.url == observation.next_url:
                step_action = action

        # save copy of the model for subsequent updates
        model_copy = copy_model(model, base_model, lora_config)

        # update the model
        model = update_policy(
            optimizer,
            tokenizer,
            objective,
            training_args,
            model,
            previous_model,
            reference_model,
            action_batch,
            rewards,
        )
        print(f"step {step}, rewards: {rewards}")

        # set previous model for the next update
        previous_model = model_copy

        if step_action is not None:
            # If the action batch contains at least one item with the correct
            # next-page target, take a step. Otherwise, don't change the state.
            observation, env_reward, terminated, truncated, info = env.step(step_action)
            if terminated or truncated:
                rprint(f"Episode terminated or truncated at step {step}")
                break

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainingArgs)
    training_args, *_ = parser.parse_args_into_dataclasses()
    main(training_args)


# TODO:
# - ✅ return the completions as an action
# - ✅ implement batch-level completions in the Agent interface
# - ✅ evaluate batch rewards and advantages
# - implement batch env.step, have to introduce storing batch states
#   in the environment
# - for each action in the batch, update environment states
# - need to handle trajectories in the batch that are already terminated
#   - just preserve the shape of the inputs but ignore the
#     actions at index position of terminated trajectories.

# IDEA:
# We can simplify the training process by computing rewards against
# the correct next step, as opposed to waiting n turns for all actions
# in the batch to reach (a) the target, or (b) hit max token configuration.
#
# The heuristic would be:
#
# - Suppose a trajectory: ["page_0", "page_1", "page_2"]
# - Assign target to "page_n"
# - Roll out trajectories for batch size of `b`
# - Compute reward from trajectories
# - Do backwards pass
# - If the "next_page" target is proposed by one sample:
#   - Assign target to "page_{n+1}"
#   - Take a step in the environment to the next pagepage
