import textwrap

import rich.markup
from rich import print as rprint
from rich.panel import Panel

from aigym.types import Action, Observation


def print_observation(observation: Observation):
    rprint(
        Panel.fit(
            textwrap.dedent(
                f"""
                [bold]URL[/bold]: {observation.url}
                [bold]Target URL[/bold]: {observation.target_url}
                [bold]Current position[/bold]: {observation.current_chunk} / {observation.total_chunks}
                """
            ),
            title="Observation",
            border_style="slate_blue3",
        )
    )


def print_context(observation: Observation, head: int = 500, tail: int = 500):
    context = observation.context[:head] + "\n...\n" + observation.context[-tail:]
    rprint(Panel.fit(rich.markup.escape(context), title="Context", border_style="yellow"))


def print_action(action: Action, index: int | None = None):
    if action.action is None:
        msg = f"Invalid completion:\n {action.completion[:500]}"
        color = "red"
    else:
        msg = textwrap.dedent(
            f"""
            [bold]Action[/bold]: {action.action}
            [bold]URL[/bold]: {action.url}
            [bold]Reasoning[/bold]: {action.reason_summary}
            """
        ).strip()
        color = "green"

    rprint(Panel.fit(msg, title=f"Action {index}" if index is not None else "Action", border_style=color))
