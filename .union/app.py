"""Simple vllm-deepseek example.

Usage:

union deploy apps app.py vllm-deepseek-test-0

"""

import union
from flytekit.extras.accelerators import A100, GPUAccelerator
from union.app import App

L40S = GPUAccelerator("nvidia-l40s")

app1 = App(
    name="vllm-deepseek-test-0",
    container_image="docker.io/vllm/vllm-openai:latest",
    command=[],
    args=[
        "--model",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "--trust-remote-code",
    ],
    port=8000,
    limits=union.Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    requests=union.Resources(cpu="2", mem="24Gi", gpu="1", ephemeral_storage="20Gi"),
    accelerator=A100,
    env={
        "DEBUG": "1",
        "LOG_LEVEL": "DEBUG",
    },
    requires_auth=False,
    scaledown_after=500,
)
