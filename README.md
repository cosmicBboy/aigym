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
uv venv
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
