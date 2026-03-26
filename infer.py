"""
Autoresearch inference research script. Single-GPU, single-file.

The base model (Qwen2.5-7B-Instruct) is fixed — no fine-tuning allowed.

Agent instructions:

- Modify the INFERENCE STRATEGY section (solve() + hyperparameters).
- Goal: maximize `score` = correct GSM8K problems solved within 5-minute budget.
- The baseline is greedy decoding with no chain-of-thought.
- Beat it by improving accuracy, speed, or both.

Usage: uv run infer.py
"""

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prepare import evaluate, extract_answer  # noqa: F401 — extract_answer exported for agents to use


console = Console()

# ---------------------------------------------------------------------------
# Hyperparameters (edit freely)
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 256  # generation budget per problem; lower = faster, higher = more reasoning room
DO_SAMPLE = False  # greedy decoding by default
TEMPERATURE = 1.0  # only applies if DO_SAMPLE = True

# ---------------------------------------------------------------------------
# INFERENCE STRATEGY — agents modify this section
# ---------------------------------------------------------------------------


def solve(model, tokenizer, problem: str) -> str:  # type: ignore[no-untyped-def]
    """
    Solve a GSM8K math problem and return a response string.

    Parameters
    ----------
    model :
        Qwen2.5-7B-Instruct loaded on CUDA in bfloat16. Weights are fixed.
    tokenizer :
        Corresponding tokenizer.
    problem : str
        Plain-English math word problem.

    Returns
    -------
    str
        Response string. The scorer extracts the LAST number, so chain-of-thought
        is fine as long as the final answer is a number.

    Notes
    -----
    Baseline: single greedy forward pass, no system prompt, no chain-of-thought.

    Ideas to try:

    - Add a chain-of-thought system prompt ("think step by step")
    - Reduce MAX_NEW_TOKENS for simpler problems (faster)
    - Self-consistency: generate N samples, take majority vote
    - Speculative decoding: use a small draft model + verify with this model
    - Quantize to int8/int4 via bitsandbytes for faster generation
    - Early stopping: detect answer token and stop immediately
    - Batch multiple problems per forward pass (set tokenizer.padding_side = "left")
    """
    messages = [{"role": "user", "content": problem}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Entry point (do not modify)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate(solve)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold")

    table.add_row("score", f"[green]{results['score']}[/green]")
    table.add_row("accuracy", f"{results['accuracy']:.4f}")
    table.add_row("tokens_per_sec", f"{results['tokens_per_sec']:.1f}")
    table.add_row("problems_attempted", str(results["problems_attempted"]))
    table.add_row("time_elapsed", f"{results['time_elapsed']:.1f}s")

    console.print()
    console.print(Panel(table, title="[bold cyan]results[/bold cyan]", expand=False))

    # Also print machine-parseable summary for grep in the experiment loop
    print("---")
    print(f"score:              {results['score']}")
    print(f"accuracy:           {results['accuracy']:.4f}")
    print(f"tokens_per_sec:     {results['tokens_per_sec']:.1f}")
    print(f"problems_attempted: {results['problems_attempted']}")
    print(f"time_elapsed:       {results['time_elapsed']:.1f}")
