"""
autoresearch agent — autonomous inference research driver.

Usage:
    uv run agent.py [--tag TAG] [--max-turns N]

Creates branch autoresearch/<tag>, runs the experiment loop via Claude Opus 4.6,
and merges back to master if the session's best score beats master's best.
"""

import argparse
import contextlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    CLINotFoundError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    query,
)
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table


console = Console()
REPO_ROOT = Path(__file__).parent

# Keywords that make agent text worth surfacing to the user
_INTERESTING = {
    "score",
    "accuracy",
    "keep",
    "discard",
    "crash",
    "baseline",
    "experiment",
    "improvement",
    "tok/s",
    "error",
    "failed",
    "✓",
    "✗",
    "→",
    "running",
    "result",
}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git(*args: str, check: bool = True) -> str:
    """Run a git command in the repo root and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_branch() -> str:
    """Return the name of the currently checked-out git branch."""
    return git("rev-parse", "--abbrev-ref", "HEAD")


# ---------------------------------------------------------------------------
# results.tsv helpers
# ---------------------------------------------------------------------------


def read_results(path: Path) -> list[dict]:
    """Parse results.tsv and return a list of row dicts (empty if file missing)."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        parts += [""] * max(0, len(headers) - len(parts))
        rows.append(dict(zip(headers, parts)))
    return rows


def best_kept(rows: list[dict]) -> tuple[int | None, str | None]:
    """Return (score, description) of the highest-scoring kept experiment."""
    kept = [r for r in rows if r.get("status") == "keep"]
    if not kept:
        return None, None
    best = max(kept, key=lambda r: int(r.get("score") or 0))
    return int(best["score"]), best.get("description", "")


# ---------------------------------------------------------------------------
# best.json helpers
# ---------------------------------------------------------------------------


def read_best_json_from_main() -> dict:
    """Read best.json from master branch (not working tree)."""
    try:
        raw = git("show", "master:best.json")
        return json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {
            "baseline_score": None,
            "best_score": None,
            "best_description": None,
            "best_branch": None,
            "updated_at": None,
        }


def write_best_json(data: dict) -> None:
    """Serialise data to best.json in the repo root."""
    (REPO_ROOT / "best.json").write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_prompt(tag: str, master_baseline: int | None, master_best: int | None) -> str:
    """Build the kickoff prompt for the Claude agent with current score targets."""
    baseline_str = str(master_baseline) if master_baseline is not None else "not yet established"
    best_str = str(master_best) if master_best is not None else "not yet established"
    date_str = datetime.now().strftime("%Y-%m-%d")

    return f"""You are running an autonomous inference research session tagged `{tag}` (date: {date_str}).

## Your task

Read `program.md` for full instructions. Key points:

- You are on branch `autoresearch/{tag}`.
- Modify only `infer.py` to improve the GSM8K score.
- Run experiments: `uv run infer.py > run.log 2>&1`
- Extract results: `grep "^score:\\|^accuracy:\\|^tokens_per_sec:" run.log`
- Log to `results.tsv` (tab-separated, columns: commit score accuracy tokens_per_sec status description).
- Kept experiments: git commit stays. Discarded: `git reset --hard HEAD~1`.

## Score targets to beat

- Master baseline (unmodified infer.py): **{baseline_str}**
- Master best score so far: **{best_str}**

Beat master best to have changes merged back to master automatically.

## Start

Read `program.md` → `prepare.py` → `infer.py`, then run the baseline first (no modifications).
NEVER stop or ask for confirmation. Run until interrupted."""


# ---------------------------------------------------------------------------
# Merge session improvements to main
# ---------------------------------------------------------------------------


def apply_to_main(
    tag: str,
    session_score: int,
    session_description: str,
    main_data: dict,
) -> bool:
    """Copy HEAD infer.py from session branch to master, update best.json."""
    original_branch = current_branch()
    try:
        # Grab infer.py content from the tip of the session branch
        infer_content = git("show", f"autoresearch/{tag}:infer.py")

        git("checkout", "master")

        (REPO_ROOT / "infer.py").write_text(infer_content)

        updated = {
            **main_data,
            "best_score": session_score,
            "best_description": session_description,
            "best_branch": f"autoresearch/{tag}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_best_json(updated)

        git("add", "infer.py", "best.json")
        msg = f"autoresearch/{tag}: {session_description} (score: {session_score})"
        git("commit", "-m", msg)

        console.print(f"[green]✓[/green] Merged to master: [italic]{msg}[/italic]")
        return True

    except Exception as exc:
        console.print(f"[red]✗[/red] Failed to apply to master: {exc}")
        return False

    finally:
        git("checkout", original_branch, check=False)


def record_baseline(tag: str, baseline_score: int, main_data: dict) -> None:
    """Write baseline_score into best.json on master (first-ever session only)."""
    original_branch = current_branch()
    try:
        git("checkout", "master")
        updated = {
            **main_data,
            "baseline_score": baseline_score,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_best_json(updated)
        git("add", "best.json")
        git("commit", "-m", f"autoresearch/{tag}: record baseline score ({baseline_score})")
        console.print(f"[green]✓[/green] Recorded baseline: {baseline_score}")
    except Exception as exc:
        console.print(f"[yellow]⚠[/yellow] Could not record baseline: {exc}")
    finally:
        git("checkout", original_branch, check=False)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------


async def run_agent(tag: str, max_turns: int, main_data: dict) -> None:
    """Launch Claude Opus 4.6 agent and stream filtered output until completion."""
    prompt = build_prompt(
        tag,
        main_data.get("baseline_score"),
        main_data.get("best_score"),
    )

    console.print(
        Panel(
            f"[bold cyan]autoresearch agent[/bold cyan]\n\n"
            f"  tag         [yellow]{tag}[/yellow]\n"
            f"  branch      [dim]autoresearch/{tag}[/dim]\n"
            f"  model       claude-opus-4-6\n"
            f"  max turns   {max_turns}\n"
            f"  baseline    [dim]{main_data.get('baseline_score') or 'not established'}[/dim]\n"
            f"  master best [dim]{main_data.get('best_score') or 'not established'}[/dim]",
            title="[bold]session start[/bold]",
            expand=False,
        )
    )

    options = ClaudeAgentOptions(
        cwd=str(REPO_ROOT),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model="claude-opus-4-6",
        system_prompt=(
            "You are an expert ML engineer autonomously researching inference optimizations. "
            "Be methodical and scientific. Write clean, minimal code. "
            "Never ask for permission or confirmation — act autonomously."
        ),
        setting_sources=[],  # ignore project CLAUDE.md so program.md drives everything
    )

    turn = 0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            sid = message.data.get("session_id", "")
            console.print(f"[dim]session: {sid}[/dim]\n")

        elif isinstance(message, AssistantMessage):
            turn += 1
            for block in message.content:
                if not isinstance(block, TextBlock):
                    continue
                lines = block.text.splitlines()
                interesting = [ln for ln in lines if any(kw in ln.lower() for kw in _INTERESTING)]
                if interesting:
                    ts = datetime.now().strftime("%H:%M:%S")
                    for ln in interesting:
                        console.print(f"[dim]{ts}[/dim]  {ln}")

        elif isinstance(message, ResultMessage):
            status = "error" if message.is_error else "ok"
            cost = f"  cost: ${message.total_cost_usd:.4f}" if message.total_cost_usd else ""
            console.print(f"\n[dim]agent finished — turns: {message.num_turns}  status: {status}{cost}[/dim]")


# ---------------------------------------------------------------------------
# Post-session summary and optional merge
# ---------------------------------------------------------------------------


def post_session(tag: str, main_data: dict) -> None:
    """Print session summary and merge to master if the session beat the best score."""
    rows = read_results(REPO_ROOT / "results.tsv")
    session_score, session_desc = best_kept(rows)

    main_best = main_data.get("best_score")
    main_baseline = main_data.get("baseline_score")

    kept = [r for r in rows if r.get("status") == "keep"]
    discarded = [r for r in rows if r.get("status") == "discard"]
    crashed = [r for r in rows if r.get("status") == "crash"]

    improved = session_score is not None and (main_best is None or session_score > main_best)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("experiments", str(len(rows)))
    table.add_row("  kept", f"[green]{len(kept)}[/green]")
    table.add_row("  discarded", f"[yellow]{len(discarded)}[/yellow]")
    table.add_row("  crashed", f"[red]{len(crashed)}[/red]")
    table.add_row("", "")
    table.add_row("master baseline", str(main_baseline) if main_baseline is not None else "[dim]—[/dim]")
    table.add_row("master best", str(main_best) if main_best is not None else "[dim]—[/dim]")
    table.add_row("session best", f"[cyan]{session_score}[/cyan]" if session_score is not None else "[dim]—[/dim]")
    table.add_row("", "")

    if improved and session_score is not None and session_desc is not None:
        delta = session_score - (main_best or 0)
        table.add_row("outcome", f"[green]IMPROVED +{delta} → merging to master[/green]")
    else:
        table.add_row("outcome", "[yellow]no improvement[/yellow]")

    console.print()
    console.print(Panel(table, title="[bold]session summary[/bold]", expand=False))

    if improved and session_score is not None and session_desc is not None:
        apply_to_main(tag, session_score, session_desc, main_data)

    # Record baseline if this was the very first session
    if main_baseline is None and rows:
        first_kept = next((r for r in rows if r.get("status") == "keep"), None)
        if first_kept:
            with contextlib.suppress(ValueError, TypeError):
                record_baseline(tag, int(first_kept["score"]), main_data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; --tag defaults to today's date (e.g. 'mar26')."""
    today = datetime.now().strftime("%b%d").lower()  # e.g. "mar26"
    parser = argparse.ArgumentParser(
        description="autoresearch — autonomous inference research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default=today,
        help=f"run tag used as branch suffix, e.g. 'mar26' → autoresearch/mar26 (default: {today})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="maximum agent turns before stopping (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: create session branch, run agent, evaluate and optionally merge."""
    args = parse_args()
    tag = args.tag
    branch = f"autoresearch/{tag}"

    console.print(Rule("[bold cyan]autoresearch[/bold cyan]"))

    # Preflight: must be on main or the target session branch
    cb = current_branch()
    if cb == "master":
        try:
            git("checkout", "-b", branch)
            console.print(f"[green]✓[/green] Created branch [bold]{branch}[/bold]")
        except subprocess.CalledProcessError:
            console.print(
                f"[red]✗[/red] Branch [bold]{branch}[/bold] already exists.\n"
                f"    Pass [bold]--tag[/bold] with a unique name, or delete the branch first."
            )
            sys.exit(1)
    elif cb == branch:
        console.print(f"[dim]Resuming on existing branch {branch}[/dim]")
    else:
        console.print(
            f"[red]✗[/red] Must run from [bold]master[/bold] or [bold]{branch}[/bold]. Currently on: [bold]{cb}[/bold]"
        )
        sys.exit(1)

    # Read main's persistent best before the agent changes anything
    main_data = read_best_json_from_main()

    # Run the agent
    try:
        anyio.run(run_agent, tag, args.max_turns, main_data)
    except CLINotFoundError:
        console.print("[red]✗[/red] Claude Code CLI not found. Run: pip install claude-agent-sdk")
        sys.exit(1)
    except CLIConnectionError as exc:
        console.print(f"[red]✗[/red] Connection error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠[/yellow] Interrupted.")

    # Evaluate and optionally merge
    post_session(tag, main_data)


if __name__ == "__main__":
    main()
