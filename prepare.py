"""
Fixed infrastructure for autoresearch inference experiments.

Handles model loading, WikiText-2 dataset, chunk preparation, and the
throughput evaluation harness.
Do not modify — this is the fixed ground truth.

Usage (one-time setup):
    uv run prepare.py

Downloads the configured model and caches the WikiText-2 test set.
"""

import math
import os
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
from transformers import logging as hf_logging


hf_logging.set_verbosity_error()
console = Console()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("AUTORESEARCH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TIME_BUDGET = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "300"))  # default 5 min
CHUNK_TOKENS = int(os.environ.get("AUTORESEARCH_CHUNK_TOKENS", "512"))

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
    """Load the configured model from cache onto the available device."""
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
# Dataset — WikiText-2
# ---------------------------------------------------------------------------


def load_chunks(tokenizer) -> list[dict]:
    """Load WikiText-2 raw test set and split into fixed-size token chunks.

    Returns a list of dicts, each with:
        ids        list[int]   token IDs (length == CHUNK_TOKENS)
        utf8_bytes int         byte length of the decoded text
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE_DIR)
    full_text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)

    chunks = []
    for start in range(0, len(all_ids) - CHUNK_TOKENS + 1, CHUNK_TOKENS):
        ids = all_ids[start : start + CHUNK_TOKENS]
        utf8_bytes = len(tokenizer.decode(ids).encode("utf-8"))
        chunks.append({"ids": ids, "utf8_bytes": utf8_bytes})
    return chunks


# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------


def evaluate(infer_fn, batch_size: int = 1) -> dict:
    """Run the fixed evaluation harness. Do not modify.

    Calls:
        infer_fn(model, tokenizer, chunks: list[list[int]]) -> list[float]

    where each float is the sum of log-probabilities (nats) for the tokens in
    that chunk.  The harness passes `batch_size` chunks per call until the
    TIME_BUDGET is exhausted.

    BPB is computed as:
        bpb = total_NLL_nats / ln(2) / total_utf8_bytes

    The clock starts AFTER model loading.  Chunks are iterated in fixed order
    for reproducibility.  Exceptions from infer_fn skip the batch (not counted).

    Returns
    -------
    dict
        tokens_per_sec   float   input tokens processed / elapsed seconds  (PRIMARY METRIC)
        bpb              float   bits per byte across all processed chunks  (QUALITY GUARD)
        chunks_processed int     chunks fully processed within budget
        time_elapsed     float   wall-clock seconds elapsed
    """
    model, tokenizer = load_model_and_tokenizer()
    chunks = load_chunks(tokenizer)

    all_ids = [c["ids"] for c in chunks]
    all_bytes = [c["utf8_bytes"] for c in chunks]
    n_chunks = len(chunks)

    total_tokens = 0
    total_nll_nats = 0.0
    total_utf8_bytes = 0
    chunks_processed = 0

    console.print(
        f"[bold]Starting evaluation[/bold]  "
        f"[dim]budget: {TIME_BUDGET}s · {n_chunks} chunks · {CHUNK_TOKENS} tokens/chunk[/dim]"
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

        i = 0
        while i < n_chunks:
            elapsed = time.time() - t_start
            if elapsed >= TIME_BUDGET:
                break

            batch_ids = all_ids[i : i + batch_size]
            batch_bytes = sum(all_bytes[i : i + batch_size])
            batch_tokens = sum(len(ids) for ids in batch_ids)

            try:
                log_probs = infer_fn(model, tokenizer, batch_ids)
                total_nll_nats += -sum(log_probs)
                total_tokens += batch_tokens
                total_utf8_bytes += batch_bytes
                chunks_processed += len(batch_ids)
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] infer_fn raised [red]{type(e).__name__}[/red] on chunk {i}: {e}")
            i += batch_size

            elapsed = time.time() - t_start
            tps = total_tokens / elapsed if elapsed > 0 else 0.0
            bpb = (total_nll_nats / math.log(2) / total_utf8_bytes) if total_utf8_bytes > 0 else float("inf")
            desc = (
                f"[cyan]{chunks_processed}[/cyan][dim]/{n_chunks}[/dim] chunks  "
                f"[cyan]{tps:.0f}[/cyan] tok/s  "
                f"bpb [bold]{bpb:.4f}[/bold]"
            )
            progress.update(task, completed=min(elapsed, TIME_BUDGET), description=desc)
            if not console.is_terminal:
                remaining = max(0, TIME_BUDGET - elapsed)
                print(
                    f"  [{int(elapsed):3d}s/{TIME_BUDGET}s | {int(remaining):3d}s left]"
                    f"  {chunks_processed} chunks  {tps:.0f} tok/s  bpb {bpb:.4f}",
                    flush=True,
                )

    time_elapsed = time.time() - t_start
    tokens_per_sec = total_tokens / time_elapsed if time_elapsed > 0 else 0.0
    bpb = (total_nll_nats / math.log(2) / total_utf8_bytes) if total_utf8_bytes > 0 else float("inf")

    return {
        "tokens_per_sec": tokens_per_sec,
        "bpb": bpb,
        "chunks_processed": chunks_processed,
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
    _, tokenizer = load_model_and_tokenizer()
    console.print()

    console.print("[bold]Step 2:[/bold] Caching WikiText-2 test set...")
    with console.status("[cyan]Fetching WikiText-2...[/]", spinner="dots"):
        chunks = load_chunks(tokenizer)
    console.print(f"[green]✓[/green] WikiText-2 cached  [dim]{len(chunks)} chunks × {CHUNK_TOKENS} tokens[/dim]\n")

    console.print(
        Panel(
            "[green]Setup complete![/green]\n\nRun experiments with:  [bold cyan]uv run infer.py[/bold cyan]",
            title="[bold]autoresearch[/bold]",
            expand=False,
        )
    )
