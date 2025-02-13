"""Example usage of the Web Gym environment."""

import textwrap

import rich.markup
import tiktoken
from webgym.agent import WebAgent
from webgym.env import WebGymEnv
from webgym.types import Action, Observation
from rich import print as rprint
from rich.panel import Panel


def print_observation(observation: Observation):
    rprint(
        Panel.fit(
            textwrap.dedent(
                f"""
                [bold]URL[/bold]: {observation.url}
                [bold]Current position[/bold]: {observation.current_chunk} / {observation.total_chunks}
                """    
            ),
            title="Observation",
            border_style="slate_blue3"
        )
    )


def print_context(observation: Observation):
    rprint(
        Panel.fit(
            rich.markup.escape(observation.context),
            title="Context",
            border_style="yellow"
        )
    )


def print_action(action: Action):
    rprint(
        Panel.fit(
            textwrap.dedent(
                f"""
                [bold]Action[/bold]: {action.action}
                [bold]URL[/bold]: {action.url}
                [bold]Reasoning[/bold]: {action.reason_summary}
                """
            ).strip(),
            title="Action",
            border_style="green"
        )
    )


def main():
    env = WebGymEnv(
        start_url="https://en.wikipedia.org/wiki/Mammal",
        web_graph_kwargs={
            "lines_per_chunk": 200,
            "overlap": 0,
        }
    )

    enc = tiktoken.get_encoding("cl100k_base")
    agent = WebAgent("deepseek-r1:7b", token_encoder=enc, n_retries_per_action=100)

    observation, info = env.reset(seed=42)
    rprint(f"reset current page to: {observation.url}")

    for step in range(1, 101):
        print_observation(observation)
        print_context(observation)
        action = agent.act(observation)
        print_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rprint(
            f"Next observation: {observation.url}, "
            f"position {observation.current_chunk} / {observation.total_chunks}"
        )
        input("Press Enter to continue...")
        if terminated or truncated:
            rprint(Panel.fit(f"Episode terminated or truncated at step {step}", border_style="spring_green3"))
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
