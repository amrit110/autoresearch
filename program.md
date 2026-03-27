# autoresearch

This is an experiment to have an LLM autonomously research inference improvements.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master (default branch).
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, model loading, WikiText-2 dataset, evaluation harness. Do not modify.
   - `infer.py` — the file you modify. Inference strategy and batching.
4. **Verify setup**: Check that the model is cached (`~/.cache/autoresearch/`). If not, tell the human to run `uv run prepare.py` first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs for a **fixed time budget of 5 minutes** (wall clock, excluding model loading). Launch it as: `uv run infer.py`.

**What you CAN do:**
- Modify `infer.py` freely — any inference-time technique is in scope. There are no hints and no prescribed approach. Reason from first principles about what might improve throughput.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It owns the fixed evaluation harness, model loading, and the ground-truth scoring logic.
- Fine-tune or update the model weights. The model is fixed (see `MODEL_ID` in `prepare.py`).
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal: maximize `tokens_per_sec`** — WikiText-2 tokens scored per wall-clock second within the 5-minute budget.

**Quality guard: `bpb`** (bits per byte) must not degrade significantly from the baseline. If `bpb` rises meaningfully, the inference is producing wrong scores — discard the experiment.

**VRAM** is a soft constraint. The active model and available hardware are shown in the agent's startup panel and in the kickoff prompt — check there for exact figures.

**Simplicity criterion**: All else being equal, simpler is better. Removing complexity and keeping the same throughput is always a win.

**The first run**: Your very first run should always establish the baseline — run `infer.py` as-is without modification.

## Output format

Once the script finishes it prints a summary like this:

```
---
tokens_per_sec:   1240.5
bpb:              4.8312
chunks_processed: 620
time_elapsed:     300.1
```

Extract the key metrics from the log:

```
grep "^tokens_per_sec:\|^bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	tokens_per_sec	bpb	status	description
```

1. git commit hash (short, 7 chars)
2. tokens_per_sec (e.g. 1240.5)
3. bpb (e.g. 4.8312)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	tokens_per_sec	bpb	status	description
a1b2c3d	1240.5	4.8312	keep	baseline
b2c3d4e	2180.3	4.8401	keep	batch_size=8 padded batching
c3d4e5f	2410.1	5.2100	discard	batch_size=16 bpb degraded
d4e5f6g	0.0	inf	crash	custom kernel OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `infer.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run infer.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Read out the results: `grep "^tokens_per_sec:\|^bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't fix it after a few attempts, give up and move on.
7. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked by git)
8. If `tokens_per_sec` improved (higher) and `bpb` did not degrade significantly, you "advance" the branch, keeping the git commit
9. If throughput is equal or worse, or bpb degraded, git reset back to where you started

**Timeout**: Model loading takes ~30s (not counted). Each experiment should finish in ~5 min + ~1 min overhead. If a run exceeds 10 minutes total, kill it and treat as a crash.

**Crashes**: If a run crashes (OOM, bug, etc.) use your judgment. Easy fix (typo, missing import)? Fix and re-run. Fundamentally broken idea? Log "crash", skip it, move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep. You are autonomous. If you run out of ideas, think harder — reason from first principles about what affects inference speed. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes ~6 minutes total, so you can run approximately 10/hour and ~80 experiments overnight. The user wakes up to a log of inference improvements.
