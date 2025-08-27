import flyte

from examples.agent_training import TrainingArgs, main

image = (
    flyte.Image.from_debian_base(name="aigym-agent-training")
    # .with_uv_project(
    #     pyproject_file=pathlib.Path("pyproject.toml"),
    #     extra_args="--extra peft --extra flyte --extra wandb --extra trl",
    #     pre=True,
    # )
    .with_pip_packages(
        "beautifulsoup4",
        "html2text",
        "httpx",
        "gymnasium",
        "markdown",
        "markdownify",
        "numpy",
        "pydantic",
        "pygame",
        "python-dotenv",
        "rich",
        "tiktoken",
        "wandb",
        "peft",
        "transformers",
        "torch>=2.7.0",
        "bitsandbytes",
        "flyte==2.0.0b17",
    )
)


env = flyte.TaskEnvironment(
    name="aigym-agent-training",
    resources=flyte.Resources(
        cpu="16",
        memory="64Gi",
        gpu="L40s:4",
    ),
    image=image,
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
