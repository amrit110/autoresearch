# autoresearch

Autonomous LLM-driven inference optimization. An AI agent modifies a single inference script, benchmarks it, keeps improvements, and loops — unattended.

## What it does

The agent runs on a dedicated git branch and iterates the experiment loop autonomously:

1. Modify `infer.py` with an inference idea
2. Run the WikiText-2 throughput benchmark (5-minute budget)
3. If `tokens_per_sec` improved and `bpb` didn't degrade → keep, advance
4. Otherwise → discard, reset, try again
5. At the end of the session, merge improvements back to `master`

You start the agent and walk away. Each experiment takes ~6 minutes; overnight you get ~80 runs.

## Metric

**Primary:** `tokens_per_sec` — WikiText-2 tokens scored per wall-clock second. Maximize this.

**Guard:** `bpb` (bits per byte) — measures output quality. Must not degrade from the baseline. If `bpb` rises, the inference is computing incorrect scores; discard the experiment.

Using a fixed corpus for throughput measurement means prompt tricks and task-specific heuristics have zero effect — only genuine inference speed improvements count.

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download model and cache dataset (one-time)
AUTORESEARCH_MODEL="Qwen/Qwen2.5-7B-Instruct" uv run prepare.py

# 4. Run a single experiment manually
uv run infer.py

# 5. Visualize results
uv run progress.py && open progress.html
```

## Running the agent

```bash
AUTORESEARCH_MODEL="Qwen/Qwen2.5-7B-Instruct" \
AUTORESEARCH_HARDWARE="L40 48GB" \
uv run agent.py
```

The agent will propose a session tag, create a branch, read the in-scope files, and start experimenting. Interrupt it with `Ctrl-C` when you want to stop.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AUTORESEARCH_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID |
| `AUTORESEARCH_HARDWARE` | `unknown` | Hardware description (shown in agent prompt) |
| `AUTORESEARCH_TIME_BUDGET` | `300` | Seconds per experiment run |
| `AUTORESEARCH_CHUNK_TOKENS` | `512` | Tokens per WikiText-2 chunk |

## Project structure

```
prepare.py      fixed: model loading, WikiText-2 dataset, evaluation harness
infer.py        agent-modified: inference strategy, BATCH_SIZE hyperparameter
agent.py        orchestration: experiment loop, git operations, session tracking
program.md      agent instructions (read by the agent at session start)
progress.py     visualization: generates progress.html from experiments.tsv
experiments.tsv cumulative log of all experiments across sessions (git-tracked)
results.tsv     per-session log written by the agent (not git-tracked)
```

## How `infer.py` works

The harness calls `infer(model, tokenizer, chunks)` where `chunks` is a list of pre-tokenized WikiText-2 windows. The function returns per-chunk log-probability sums (nats). The baseline is a simple serial forward pass; the agent is free to implement batching, quantization, kernel tricks, or anything else.

```python
BATCH_SIZE = 1  # agent sets this; harness passes BATCH_SIZE chunks per call

def infer(model, tokenizer, chunks: list[list[int]]) -> list[float]:
    # return sum of log P(token_t | token_{<t}) for each chunk
    ...
```

## Viewing progress

```bash
uv run progress.py        # writes progress.html
open progress.html
```

The report shows tokens/sec per experiment (color-coded by status), a running-best line, BPB scatter, and the full experiment log.

## License

MIT
