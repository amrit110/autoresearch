"""
Fixed infrastructure for autoresearch inference experiments.

Handles model loading, GSM8K dataset, and the evaluation harness.
Do not modify — this is the fixed ground truth.

Usage (one-time setup):
    uv run prepare.py

Downloads Qwen2.5-7B-Instruct and caches GSM8K test set.
"""

import os
import re
import time

import torch
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


console = Console()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("AUTORESEARCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "300"))  # default 5 min

# Device + dtype: CUDA on Linux/GPU servers, MPS on Apple Silicon, CPU fallback
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16  # bfloat16 has limited MPS support
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer():
    """Load Qwen2.5-7B-Instruct from cache onto CUDA in bfloat16."""
    with console.status(f"[bold cyan]Loading {MODEL_ID}...[/]", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        model.eval()
    if DEVICE == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        mem_str = f"VRAM: {mem_gb:.1f} GB"
    else:
        mem_str = f"device: {DEVICE}"
    console.print(f"[green]✓[/green] Model loaded  [dim]{mem_str}[/dim]")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_problems():
    """Load GSM8K test set. Returns list of {'question': str, 'answer': str}."""
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=CACHE_DIR)
    return [{"question": row["question"], "answer": row["answer"]} for row in dataset]


# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------


def extract_answer(text):
    """
    Extract the last number from a model response.

    Returns normalized string (e.g. "42") or None.
    Exported so solve() authors can use it if helpful.
    """
    text = text.replace(",", "")  # strip comma separators: 1,234 -> 1234
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def score_answer(predicted, ground_truth_raw):
    """
    Parse '#### N' from ground_truth_raw and compare to predicted.

    Returns True iff they match numerically.
    """
    match = re.search(r"####\s*([^\n]+)", ground_truth_raw)
    if not match:
        return False
    gt = match.group(1).strip().replace(",", "")
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gt)) < 1e-6
    except ValueError:
        return predicted.strip() == gt.strip()


# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


def evaluate(solve_fn):
    """
    Run the fixed evaluation harness. Do not modify.

    Loads model + tokenizer once, then calls solve_fn(model, tokenizer, problem)
    on GSM8K test problems until the TIME_BUDGET (5 min) is exhausted.

    The clock starts AFTER model loading — load time is not counted.
    Problems are always iterated in the same fixed order for reproducibility.
    If solve_fn raises an exception, that problem counts as wrong and the loop continues.
    In-flight problems when the budget expires are discarded (not counted).

    Returns
    -------
    dict
        score               int    correct problems solved (PRIMARY METRIC — maximize this)
        accuracy            float  score / problems_attempted
        tokens_per_sec      float  output tokens generated / total elapsed seconds
        problems_attempted  int    problems started and completed within budget
        time_elapsed        float  wall-clock seconds elapsed
    """
    model, tokenizer = load_model_and_tokenizer()
    problems = load_problems()

    correct = 0
    attempted = 0
    total_output_tokens = 0

    console.print(
        f"[bold]Starting evaluation[/bold]  [dim]budget: {TIME_BUDGET}s · {len(problems)} problems available[/dim]"
    )

    t_start = time.time()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/]"),
        BarColumn(bar_width=28),
        TextColumn("[dim]{task.completed:.0f}s / {task.total}s[/dim]"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("evaluating...", total=TIME_BUDGET)

        for i, prob in enumerate(problems):
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET:
                break

            question = prob["question"]
            gt_raw = prob["answer"]

            try:
                response = solve_fn(model, tokenizer, question)
                out_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                total_output_tokens += out_tokens
                predicted = extract_answer(response)
                if score_answer(predicted, gt_raw):
                    correct += 1
                attempted += 1
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] solve_fn raised [red]{type(e).__name__}[/red] on problem {i}: {e}")
                attempted += 1

            elapsed = time.time() - t_start
            acc = correct / attempted if attempted > 0 else 0.0
            tps = total_output_tokens / elapsed if elapsed > 0 else 0.0
            desc = (
                f"[green]{correct}[/green][dim]/{attempted}[/dim] correct  "
                f"acc [bold]{acc:.3f}[/bold]  "
                f"[cyan]{tps:.0f}[/cyan] tok/s"
            )
            progress.update(task, completed=min(elapsed, TIME_BUDGET), description=desc)

    time_elapsed = time.time() - t_start
    accuracy = correct / attempted if attempted > 0 else 0.0
    tokens_per_sec = total_output_tokens / time_elapsed if time_elapsed > 0 else 0.0

    return {
        "score": correct,
        "accuracy": accuracy,
        "tokens_per_sec": tokens_per_sec,
        "problems_attempted": attempted,
        "time_elapsed": time_elapsed,
    }


# ---------------------------------------------------------------------------
# Main — one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    console.rule("[bold cyan]autoresearch setup[/bold cyan]")
    console.print(f"[dim]Cache directory:[/dim] {CACHE_DIR}\n")

    console.print("[bold]Step 1:[/bold] Downloading model weights...")
    load_model_and_tokenizer()
    console.print()

    console.print("[bold]Step 2:[/bold] Caching GSM8K test set...")
    with console.status("[cyan]Fetching GSM8K...[/]", spinner="dots"):
        problems = load_problems()
    console.print(f"[green]✓[/green] GSM8K cached  [dim]{len(problems)} test problems[/dim]\n")

    console.print(
        Panel(
            "[green]Setup complete![/green]\n\nRun experiments with:  [bold cyan]uv run infer.py[/bold cyan]",
            title="[bold]autoresearch[/bold]",
            expand=False,
        )
    )
