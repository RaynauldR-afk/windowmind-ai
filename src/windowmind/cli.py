import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from .config import get_settings
from .graph import WindowMindAgent

console = Console()

def main():
    parser = argparse.ArgumentParser(description="WindowMind AI - sliding-window research agent")
    parser.add_argument("--file", required=True, help="Path to a UTF-8 text or Markdown document")
    parser.add_argument("--question", required=True, help="Question to answer from the document")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Document not found: {path}")
    document = path.read_text(encoding="utf-8")
    settings = get_settings()
    agent = WindowMindAgent(settings)

    console.print(Panel.fit("WindowMind AI | Sliding Window Research Agent", style="bold cyan"))
    result = agent.run(document, args.question)

    stats = result["stats"]
    table = Table(title="Sliding Window Telemetry")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)
    console.print(Panel(result["answer"], title="Grounded Answer", border_style="green"))
    console.print("\n[dim]Retrieved context is grounded in the indexed document windows.[/dim]")

if __name__ == "__main__":
    main()
