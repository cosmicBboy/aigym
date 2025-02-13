# Web Gymnasium

*The Web as RL environments for LM training*

This package contains constructs for converting the web into an RL environment
where pre-trained language models can be fine-tuned via RL (PPO and other
methods).

`web-gym` provides an interface for defining RL environments as mini-games
on the web and exposes a few built-in games such as the **wikipedia navigation**
game.

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
uv pip install .
```

Install `ollama` to run a local model: https://ollama.com/download

## Usage

Run the basic example:

```bash
python examples/basic_usage.py
```
