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

Examples using cli args:
```bash
python examples/agent_training.py \
    --model_id google/gemma-3-270m-it \
    --enable_gradient_checkpointing \
    --n_hops 1 \
    --n_tries_per_hop 10 \
    --rollout_min_new_tokens 64 \
    --rollout_max_new_tokens 128 \
    --group_size 4
```

With weights and biases:

```bash
export WANDB_API_KEY=...
python examples/agent_training.py \
    --model_id google/gemma-3-1b-it \
    --enable_gradient_checkpointing \
    --n_hops 1 \
    --n_tries_per_hop 10 \
    --rollout_min_new_tokens 256 \
    --rollout_max_new_tokens 512 \
    --group_size 4 \
    --wandb_project aigym-agent-training
```
"""

import tempfile
from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, PeftModelForCausalLM, get_peft_model
from rich import print as rprint
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CompileConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Qwen2ForCausalLM,
)

import aigym.pprint as pprint
import aigym.prompts as prompts
import wandb
from aigym.agent import Agent
from aigym.env import WikipediaGymEnv
from aigym.types import Action, ActionBatch, Observation, RolloutBatch


@dataclass
class TrainingArgs:
    n_episodes: int = 10
    model_id: str = "google/gemma-3-1b-it"
    ref_model_id: str | None = None
    optim: str = "adamw"
    lr: float = 1e-4
    group_size: int = 4
    advantage_eps: float = 1e-8
    clip_eps: float = 0.2
    kl_weight: float = 0.01
    max_grad_norm: float = 1.0
    use_lora: bool = True
    use_bnb_quantization: bool = False
    enable_gradient_checkpointing: bool = False
    attn_implementation: str = "eager"
    n_hops: int = 1
    n_tries_per_hop: int = 10
    rollout_min_new_tokens: int = 64
    rollout_max_new_tokens: int = 128
    rollout_temperature: float = 1.25
    rollout_top_k: int = 64
    rollout_top_p: float = 0.95
    rollout_repetition_penalty: float = 1.0
    rollout_no_repeat_ngram_size: int = 0
    wandb_project: str = None


class TrainingLogger(Protocol):
    def log_environment(self, env: WikipediaGymEnv) -> None: ...
    def log_actions(self, actions: list[Action]) -> None: ...
    def log_observation(self, observation: Observation) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_rewards(self, rewards: torch.Tensor, returns: torch.Tensor) -> None: ...
    def flush(self) -> None: ...


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
    model_copy: PreTrainedModel | PeftModelForCausalLM,
) -> PreTrainedModel:
    if isinstance(model, PeftModelForCausalLM):
        model = cast(PeftModelForCausalLM, model)
        model_copy.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
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
        log_probs_old: torch.Tensor | None,
        log_probs_ref: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        if log_probs_old is not None:
            ratio = (log_probs - log_probs_old).exp()
        else:
            ratio = log_probs

        surr1 = ratio * returns
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * returns
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Reference: http://joschu.net/blog/kl-approx.html
    """
    log_ratio = log_probs_ref - log_probs
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    return log_ratio.exp() - log_ratio - 1


def compute_log_probs(
    model: Qwen2ForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
):
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clone()  # Clone to avoid modifying input
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

    torch.cuda.empty_cache()

    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    # truncate logits to only include the action tokens
    logits = output["logits"][:, prompt_length:-1].clone()
    output_ids = sequence_ids[:, prompt_length + 1 :].clone()

    torch.cuda.empty_cache()
    del sequence_ids, output
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def reward_function(action: Action, observation: Observation) -> float:
    """Reward function.

    - no/invalid action = 0
    - is next url 0.5
    - is target url 1.0
    """
    reward = 0
    if action.action is None:
        return reward

    if action.url == observation.target_url:
        reward += 1.0
    elif action.url == observation.next_url:
        reward += 0.5

    return reward


@torch.no_grad()
def policy(
    training_args: TrainingArgs,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    prompt: str,
) -> RolloutBatch:
    # tokenize and prepare inputs for batch generation

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
    chat_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)

    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(training_args.group_size, 1)
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(training_args.group_size, 1)

    # generate completions
    model.eval()
    with torch.no_grad():
        sequence_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completions = tokenizer.batch_decode(
        sequence_ids[:, model_inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return RolloutBatch(
        sequence_ids=sequence_ids,
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        completions=completions,
    )


def reconstruct_sequence_ids(
    action_batch: ActionBatch,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstructs sequence ids from the action batch.

    The action mask is used to mask out the input id from the loss calculation.
    The attention mask is used to mask out the padding tokens from the loss calculation.
    """

    log_probs_old = None

    if action_batch.log_probs is not None:
        log_probs_old = action_batch.log_probs.detach()

    sequence_ids = action_batch.sequence_ids.detach()

    # action mask makes sure end of sequence tokens are masked out of the
    # loss calculation
    completion_ids = sequence_ids[:, action_batch.input_ids.shape[1] :]
    action_mask = torch.full_like(completion_ids, fill_value=True, dtype=torch.bool)
    action_mask[completion_ids == tokenizer.eos_token_id] = False
    action_mask = action_mask[:, 1:]

    pad_token_id = tokenizer.eos_token_id
    attention_mask = sequence_ids != pad_token_id

    return sequence_ids, action_mask, attention_mask, log_probs_old


def update_policy(
    action_batch: ActionBatch,
    returns: torch.Tensor,
    optimizer: optim.Optimizer,
    tokenizer: PreTrainedTokenizer,
    objective: GRPOLoss,
    training_args: TrainingArgs,
    model: PreTrainedModel,
    reference_model: PreTrainedModel,
):
    model.train()
    optimizer.zero_grad()

    sequence_ids, action_mask, attention_mask, log_probs_old = reconstruct_sequence_ids(
        action_batch,
        tokenizer,
    )

    # Idea: trucate or zero out the logits such that only the action tokens are
    # used in the per-token log probabilities
    prompt_length = action_batch.input_ids.shape[1]
    log_probs = compute_log_probs(
        model,
        sequence_ids,
        attention_mask,
        prompt_length=prompt_length,
    )
    with torch.no_grad():
        log_probs_ref = compute_log_probs(
            reference_model,
            sequence_ids,
            attention_mask,
            prompt_length=prompt_length,
        )

    loss, kl = objective(
        log_probs=log_probs,
        log_probs_old=log_probs_old,
        log_probs_ref=log_probs_ref,
        returns=returns,
        action_mask=action_mask,
    )

    if not loss.isfinite():
        print(f"Loss not finite, skipping backward, loss={loss}, returns: {returns}")
        del log_probs, log_probs_old, log_probs_ref, sequence_ids, action_mask, attention_mask, loss, kl
        torch.cuda.empty_cache()
        return model

    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=training_args.max_grad_norm)
    print(f"loss={loss: .10f}, kl={kl: .10f}, grad_norm={grad_norm: .10f}")
    optimizer.step()
    torch.cuda.empty_cache()
    return model, loss, kl, grad_norm


class PrintLogger:
    def log_environment(self, env: WikipediaGymEnv) -> None:
        pprint.print_travel_path(env.travel_path)

    def log_actions(self, actions: list[Action]) -> None:
        for i, action in enumerate(actions):
            pprint.print_action(action, index=i)

    def log_observation(self, observation: Observation) -> None:
        pprint.print_observation(observation)
        pprint.print_context(observation)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        _metrics = {k: float(v) for k, v in metrics.items()}
        pprint.print_metrics(_metrics)

    def log_rewards(self, rewards: torch.Tensor, returns: torch.Tensor) -> None:
        pprint.print_rewards(
            {
                "Raw rewards": rewards.squeeze().tolist(),
                "Returns": returns.squeeze().tolist(),
            }
        )

    def flush(self) -> None: ...


def main(training_args: TrainingArgs, logger: TrainingLogger | None = None):
    if logger is None:
        logger = PrintLogger()

    if training_args.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=training_args.wandb_project)

    enc = tiktoken.get_encoding("cl100k_base")

    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("\nAvailable GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print()

    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_id)

    bnb_config = None
    if training_args.use_bnb_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    reference_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        training_args.ref_model_id or training_args.model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=training_args.attn_implementation,
        quantization_config=bnb_config,
    )

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        training_args.model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=training_args.attn_implementation,
        quantization_config=bnb_config,
    )

    lora_config = None
    if training_args.use_lora:
        target_lora_modules = ["q_proj", "k_proj", "v_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=target_lora_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            reference_model = get_peft_model(reference_model, lora_config, adapter_name="default")
            reference_model.save_pretrained(tmp_dir)

            model = get_peft_model(model, lora_config, adapter_name="default")
            model.load_adapter(tmp_dir, adapter_name="default", is_trainable=True)
            model.print_trainable_parameters()

        reference_model.set_adapter("default")
        model.set_adapter("default")

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
        min_new_tokens=training_args.rollout_min_new_tokens,
        max_new_tokens=training_args.rollout_max_new_tokens,
        temperature=training_args.rollout_temperature,
        top_k=training_args.rollout_top_k,
        top_p=training_args.rollout_top_p,
        repetition_penalty=training_args.rollout_repetition_penalty,
        no_repeat_ngram_size=training_args.rollout_no_repeat_ngram_size,
        padding=True,
        padding_size="left",
        return_attention_mask=True,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["<end_of_turn>"],
        compile_config=CompileConfig(fullgraph=False),
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

    env = WikipediaGymEnv(n_hops=training_args.n_hops)
    n_tries = int(training_args.n_hops * training_args.n_tries_per_hop)

    # Separate rollout step from model updates, i.e. implement an experience
    # replay buffer so that you can sample from it, then update the model.
    # This clarifies the roll of the `old_log_probs` in the loss calculation.
    # You'll have multiple rollouts, you'll sample from this buffer in minibatches
    # so therefore the current model log probs will start to deviate from the
    # old log probs.
    # See: https://www.perplexity.ai/search/can-you-give-me-the-basic-grpo-g3kdNUI4RSmbAtoxi_HcjQ#6

    total_cumulative_rewards = 0
    console = Console(highlight=False)
    for episode in range(1, training_args.n_episodes + 1):
        console.rule(f"▶️ Episode {episode}")
        observation, info = env.reset()

        logger.log_environment(env)

        n_steps = 0
        episode_cumulative_returns = 0
        episode_cumulative_rewards = 0

        for step in range(1, n_tries):
            n_steps += 1
            console.rule(f"Step {step}")
            logger.log_observation(observation)
            action_batch: ActionBatch = agent.act(observation)

            step_action: Action | None = None
            rewards: list[float] = []
            for i, action in enumerate(action_batch.actions):
                reward = reward_function(action, observation)
                rewards.append(reward)
                if action.action is None:
                    continue
                if action.url == observation.next_url:
                    step_action = action
                else:
                    rprint(f"↪️ Action {action.url} doesn't go to the next url {observation.next_url}, skipping")

            logger.log_actions(action_batch.actions)

            rewards: torch.Tensor = torch.tensor(rewards, dtype=model.dtype).unsqueeze(1)
            returns = (rewards - rewards.mean()) / (rewards.std() + training_args.advantage_eps)
            returns = returns.to(model.device)
            logger.log_rewards(rewards, returns)

            if (returns == 0).all():
                update_metrics = {}
            else:
                # update the model
                model, loss, kl, grad_norm = update_policy(
                    action_batch,
                    returns,
                    optimizer,
                    tokenizer,
                    objective,
                    training_args,
                    model,
                    reference_model,
                )
                update_metrics = {
                    "loss": loss,
                    "kl": kl,
                    "grad_norm": grad_norm,
                }

            rewards_sum = rewards.sum()
            total_cumulative_rewards += rewards_sum
            episode_cumulative_rewards += rewards_sum
            episode_cumulative_returns += returns.squeeze().sum()

            metrics = {
                "returns": returns.mean(),
                "rewards": rewards.mean(),
                "n_steps": n_steps,
                "episode_cumulative_returns": episode_cumulative_returns,
                "episode_cumulative_rewards": episode_cumulative_rewards,
                "log_total_cumulative_rewards": torch.log(total_cumulative_rewards),
                **update_metrics,
            }
            logger.log_metrics(metrics)
            wandb.log(metrics)

            if step_action is not None:
                # If the action batch contains at least one item with the correct
                # next-page target, take a step. Otherwise, don't change the state.
                observation, env_reward, terminated, truncated, info = env.step(step_action)
                if terminated or truncated:
                    rprint(f"Episode terminated or truncated at step {step}")
                    break

        logger.flush()

    rprint("Task finished!")
    env.close()


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainingArgs)
    training_args, *_ = parser.parse_args_into_dataclasses()
    main(training_args)
