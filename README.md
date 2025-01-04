# Web Gym

*The Web as RL environments for LM training*

This package contains constructs for converting the web into an RL environment
where pre-trained language models can be fine-tuned via RL (PPO and other
methods).

`web-gym` provides an interface for defining RL environments as mini-games
on the web and exposes a few built-in games such as the **wikipedia navigation**
game.

## Development Installation

```bash
pip install .
```

Install `ollama` to run a local model: https://ollama.com/download
