import rich.markup
from rich import print as rprint
from rich.console import Console

from aigym.types import Action, Observation


def print_travel_path(travel_path: list[str]):
    console = Console(highlight=False)
    console.print("[slate_blue3]■ 🌍 Travel Path[/slate_blue3]")
    for i, value in enumerate(travel_path):
        if i == len(travel_path) - 1:
            char = "└──"
            end = "\n\n"
        else:
            char = "├──"
            end = "\n"
        console.print(f"{char} [link={value}]{value}[/link]", end=end)
    console.print(end="\n\n")


def print_observation(observation: Observation):
    rprint("[slate_blue3]■ 🌍 Observation[/slate_blue3]")
    rprint(f"├── Current URL: [link={observation.url}]{observation.url}[/link]")
    rprint(f"├── Next URL: [link={observation.next_url}]{observation.next_url}[/link]")
    rprint(f"└── Target URL: [link={observation.target_url}]{observation.target_url}[/link]", end="\n\n")


def print_context(observation: Observation, head: int = 500, tail: int = 500):
    console = Console(highlight=False)
    console.print(f"[yellow]■ Context Chunks[/yellow]: {observation.url}")

    if len(observation.chunk_names) == 0:
        console.print("└── No context chunks", end="\n\n")
        return

    for i, chunk_name in enumerate(observation.chunk_names):
        if i == len(observation.chunk_names) - 1:
            char = "└──"
            end = "\n\n"
        else:
            char = "├──"
            end = "\n"
        console.print(f"{char} [link={observation.url}#{chunk_name}]{chunk_name}[/link]", end=end)


def print_action(action: Action, index: int | None = None):
    completion_text = rich.markup.escape(
        action.completion.replace("\n", "").replace("\r", "").replace("\t", "").replace("    ", "").replace("  ", "")
    )

    console = Console(highlight=False)
    if action.action is None:
        console.print(f"[red]■ Invalid Action [{index or 0}][/red]")
        console.print(f"├── Error type: {action.error_type}")
        console.print(f"├── Parse type: {action.parse_type}")
        console.print(f"└── Completion: {completion_text}", end="\n\n")
    else:
        console.print(f"[orange3]■ Action [{index or 0}][/orange3]")
        console.print(f"├── Action: {action.action}")
        console.print(f"├── URL: {action.url}")
        console.print(f"├── Reasoning: {action.reason_summary}")
        console.print(f"├── Parse type: {action.parse_type}")
        console.print(f"└── Completion: {completion_text}", end="\n\n")


def print_rewards(rewards: dict):
    console = Console(highlight=False)
    console.print("[green]■ 💰 Rewards[/green]")
    for i, (key, value) in enumerate(rewards.items()):
        if i == len(rewards) - 1:
            char = "└──"
            end = "\n\n"
        else:
            char = "├──"
            end = "\n"
        console.print(f"{char} {key}: {value}", end=end)
    console.print(end="\n\n")


def print_metrics(metrics: dict):
    console = Console(highlight=False)
    console.print("[blue]■ 📊 Metrics[/blue]")
    for i, (key, value) in enumerate(metrics.items()):
        if i == len(metrics) - 1:
            char = "└──"
            end = "\n\n"
        else:
            char = "├──"
            end = "\n"
        console.print(f"{char} {key}: {value}", end=end)
    console.print(end="\n\n")
