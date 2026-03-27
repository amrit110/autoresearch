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

BATCH_SIZE = 600  # all chunks in one call
SUB_BATCH = 8  # internal GPU batch size

# ---------------------------------------------------------------------------
# INFERENCE STRATEGY — agents modify this section
# ---------------------------------------------------------------------------

_compiled_model = None


def _get_compiled(model):  # type: ignore[no-untyped-def]
    """Lazily compile model with inductor backend, baking in use_cache=False."""
    global _compiled_model  # noqa: PLW0603
    if _compiled_model is None:
        model.config.use_cache = False
        _compiled_model = torch.compile(model, fullgraph=True)
    return _compiled_model


def infer(model, tokenizer, chunks: list[list[int]]) -> list[float]:  # type: ignore[no-untyped-def]
    """Process all chunks in one call with compiled sub-batching."""
    compiled = _get_compiled(model)
    device = model.device
    all_ids = torch.tensor(chunks, dtype=torch.long, device=device)
    n = all_ids.shape[0]
    results = torch.empty(n, device=device)
    with torch.inference_mode():
        for i in range(0, n, SUB_BATCH):
            batch = all_ids[i : i + SUB_BATCH]
            logits = compiled(batch).logits
            nll = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                batch[:, 1:],
                reduction="none",
            )
            results[i : i + batch.shape[0]] = -nll.sum(dim=1)
    return results.tolist()


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
