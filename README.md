# 🤖 AI Gym

*Reinforcement learning environments for AI fine-tuning*

`aigym` is a library that provides a suite of reinforcement learning (RL)
environments primarily for the purpose of fine-tuning pre-trained models - namely
language models - for various reasoning tasks.

Built on top of the [gymnasium](https://gymnasium.farama.org/) API, the objective
of this project is to expose a light-weight and extensible environments
to fine-tune language models with techniques like [PPO](https://arxiv.org/abs/1707.06347)
and [GRPO](https://arxiv.org/abs/2402.03300).

It is designed to complement training frameworks like [trl](https://huggingface.co/docs/trl/en/index),
[transformers](https://huggingface.co/docs/transformers/en/index), [pytorch](https://pytorch.org/),
and [pytorch lightning](https://lightning.ai/pytorch-lightning).

See the project roadmap [here](./ROADMAP.md)

## Installation

```bash
pip install aigym
```

## Development Installation

Install `uv`:

```bash
pip install uv
```

Create a virtual environment:

```bash
uv venv --python 3.12
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the package:

```bash
uv sync --extra ollama --group dev
```

Install `ollama` to run a local model: https://ollama.com/download

## Usage

The `examples` directory contains examples on how to use the `aigym` environments.
Run an ollama-based agent on the Wikipedia maze environment:

```bash
python examples/ollama_agent.py
```

### Agent training on Flyte

Install flyte:

```bash
uv pip install '.[flyte]'
```

The `examples/agent_training.py` example can by run on a Flyte cluster using
the `examples/agent_training_flyte.py` entrypoint. First, create a configuration:

```bash
flyte create config \
--endpoint demo.hosted.unionai.cloud \
--builder remote \
--project aigym \
--domain development
```

This will create a `config.yaml` file in the current directory.

```bash
PYTHONPATH=. python examples/agent_training_flyte.py \
    --model_id google/gemma-3-1b-it \
    --enable_gradient_checkpointing \
    --n_episodes 100 \
    --n_hops 1 \
    --n_tries_per_hop 10 \
    --rollout_min_new_tokens 128 \
    --rollout_max_new_tokens 256 \
    --group_size 4 \
    --wandb_project aigym-agent-training
```
