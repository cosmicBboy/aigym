import flyte

from examples.agent_training import TrainingArgs, main

env = flyte.TaskEnvironment(
    name="aigym-agent-training",
    resources=flyte.Resources(
        cpu="4",
        memory="20Gi",
        gpu="L40s:4",
    ),
    image=(
        flyte.Image.from_debian_base(
            name="aigym-agent-training",
            python_version=(3, 12),
            flyte_version="2.0.0b17",
        ).with_uv_project("./pyproject.toml", extra_args="--extra flyte --extra wandb --extra peft")
    ),
    secrets=[
        flyte.Secret(key="huggingface_token", as_env_var="HF_TOKEN"),
        flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
    ],
)


@env.task
def agent_training(args: TrainingArgs):
    main(args)


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainingArgs)
    training_args, *_ = parser.parse_args_into_dataclasses()

    flyte.init_from_config("./config.yaml")
    run = flyte.run(agent_training, args=training_args)
    print(run.url)
