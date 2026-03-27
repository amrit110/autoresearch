"""
Autoresearch inference research script. Single-GPU, single-file.

The base model is fixed — no fine-tuning allowed.

Agent instructions:

- Modify the INFERENCE STRATEGY section (infer() + hyperparameters).
- Goal: maximize `tokens_per_sec` on WikiText-2 without degrading `bpb`.
- The baseline is a serial forward pass, one chunk at a time.
- Beat it with batching, quantization, kernel optimizations, etc.

Usage: uv run infer.py
"""

import torch
import torch.nn.functional as F  # noqa: N812
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prepare import evaluate


console = Console()

# ---------------------------------------------------------------------------
# Hyperparameters (edit freely)
# ---------------------------------------------------------------------------

BATCH_SIZE = 1  # chunks passed to infer() per call; increase for batched inference

# ---------------------------------------------------------------------------
# INFERENCE STRATEGY — agents modify this section
# ---------------------------------------------------------------------------


def infer(model, tokenizer, chunks: list[list[int]]) -> list[float]:  # type: ignore[no-untyped-def]
    """Score a batch of token-ID chunks; return per-chunk log-prob sums (nats).

    Parameters
    ----------
    model :
        Causal LM loaded on device in bfloat16. Weights are fixed.
    tokenizer :
        Corresponding tokenizer.
    chunks : list[list[int]]
        Each inner list is CHUNK_TOKENS token IDs (all equal length).

    Returns
    -------
    list[float]
        One float per chunk: sum of log P(token_t | token_{<t}) in nats.
        Higher (less negative) means the model assigns more probability to
        this text — i.e. lower perplexity.

    Notes
    -----
    Baseline: independent forward pass per chunk, no batching.
    BPB must not degrade significantly vs. baseline — if it rises,
    the scoring is wrong and the experiment should be discarded.
    """
    results = []
    for ids in chunks:
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits  # (1, T, V)
        shift_logits = logits[0, :-1, :]  # (T-1, V)
        shift_labels = input_ids[0, 1:]  # (T-1,)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs[torch.arange(len(shift_labels)), shift_labels]
        results.append(token_log_probs.sum().item())
    return results


# ---------------------------------------------------------------------------
# Entry point (do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate(infer, batch_size=BATCH_SIZE)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")

    table.add_row("tokens_per_sec", f"[cyan]{results['tokens_per_sec']:.1f}[/cyan]")
    table.add_row("bpb", f"{results['bpb']:.4f}")
    table.add_row("chunks_processed", str(results["chunks_processed"]))
    table.add_row("time_elapsed", f"{results['time_elapsed']:.1f}s")

    console.print()
    console.print(Panel(table, title="[bold cyan]results[/bold cyan]", expand=False))

    # Machine-parseable summary for grep in the experiment loop
    print("---")
    print(f"tokens_per_sec:   {results['tokens_per_sec']:.1f}")
    print(f"bpb:              {results['bpb']:.4f}")
    print(f"chunks_processed: {results['chunks_processed']}")
    print(f"time_elapsed:     {results['time_elapsed']:.1f}")
