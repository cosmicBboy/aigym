import flyte
import flyte.report
import torch

import aigym.types as types
from aigym.env import WikipediaGymEnv
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
        "flyte==2.0.0b18",
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


class ReportLogger:
    def log_environment(self, env: WikipediaGymEnv):
        html = "<h1>Environment</h1>"
        html += f"<p>{env.travel_map}</p>"
        html += f"<p>{env.travel_path}</p>"
        env_tab = flyte.report.get_tab("Environment")
        env_tab.replace(html)

    def log_actions(self, actions: list[types.Action]):
        html = ""
        for i, action in enumerate(actions):
            html += f"<h2>Action {i}</h2>"
            html += f"<p>{action}</p>"
        actions_tab = flyte.report.get_tab("Actions")
        actions_tab.replace(html)

    def log_observation(self, observation: types.Observation):
        html = "<h2>Observation</h2>"
        html += f"<p>{observation}</p>"
        html += "<h2>Context</h2>"
        html += f"<p>{observation.context}</p>"
        observation_tab = flyte.report.get_tab("Observation")
        observation_tab.replace(html)

    def log_metrics(self, metrics: dict[str, float]):
        html = "<h2>Metrics</h2>"
        html += f"<p>{metrics}</p>"
        metrics_tab = flyte.report.get_tab("Metrics")
        metrics_tab.replace(html)

    def log_rewards(self, rewards: torch.Tensor, returns: torch.Tensor):
        html = "<h2>Rewards</h2>"
        html += f"<p>{rewards.squeeze().tolist()}</p>"
        html += "<h2>Returns</h2>"
        html += f"<p>{returns.squeeze().tolist()}</p>"
        rewards_tab = flyte.report.get_tab("Rewards")
        rewards_tab.replace(html)

    def flush(self):
        flyte.report.flush()


@env.task(report=True)
def agent_training(args: TrainingArgs):
    main(args)
    # main(args, ReportLogger())


if __name__ == "__main__":
    from transformers import HfArgumentParser

    parser = HfArgumentParser(TrainingArgs)
    training_args, *_ = parser.parse_args_into_dataclasses()

    flyte.init_from_config("./config.yaml")
    run = flyte.run(agent_training, args=training_args)
    print(run.url)
