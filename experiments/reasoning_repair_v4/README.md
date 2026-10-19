# FACET reasoning supervision repair

This experiment reuses the exact 1,191 previously selected v2 source tasks. Each
assistant turn becomes a separate training example. Earlier assistant messages
retain their JSON output, while only the current target retains its full teacher
reasoning. LLaMA-Factory uses `mask_history=true`: all history labels are ignored.
The first user message stays in the raw Harbor format used during evaluation.

This changes the training unit from a complete trajectory to an assistant turn.
Three epochs and an effective batch of 64 therefore produce more optimizer steps
than the earlier 57-step run. It also restores the original user-message format.
Results are a repair experiment, not a causal estimate of reasoning alone or an
exact reproduction of the paper. Matched no-reasoning examples are generated for
a later controlled comparison but are not automatically trained.

## Locations and synchronization

Code is synchronized through the `codex/reasoning-repair-v4` branch of
`bukysun/FACET-Terminal`. The original remote checkout is
`/root/workspace/train_exp/FACET-Terminal`. Do not change or remove its untracked
jobs, data, reports, old checkpoints, or other users' workloads.

The new run uses `work/reasoning-repair-v4/`. This directory is ignored by Git.
Keep generated data, private environment configuration, outputs, logs, manifests,
and the state file there. Never commit them or credentials. The new code does not
load the old `jobs/envs/.env`; all inference is local and needs no paid API key.

## CPU preparation

Run from the project root using the installed Python environments:

```sh
python3 experiments/reasoning_repair_v4/bootstrap.py \
  --project /root/workspace/train_exp/FACET-Terminal \
  --run /root/workspace/train_exp/FACET-Terminal/work/reasoning-repair-v4
```

Preparation is exclusive and never overwrites a previous preparation state.
`prepare.py` preserves the existing source selection and records file hashes,
task IDs, target indices, exact training-template lengths, package versions, and
all exclusions. Overlength examples are dropped; the current reasoning/answer
is never silently truncated. The script encodes history once per source task
and checks cached lengths against full LLaMA-Factory rendering.

`validate_labels.py` runs 64 deterministic examples from each arm through the
actual installed training processor. It asserts that the only supervised tokens
are the complete current target, all history labels are masked, and there is no
truncation. `configure.py` creates a private benchmark snapshot, model-only
checkpoint training configs, and matched evaluation configs. `validate_configs.py`
checks the installed Harbor schema and correct `llm_call_kwargs` forwarding.

## Half-hour follow-up and resource gate

The user authorized checking GPU availability every 30 minutes and starting
training when resources are free. The associated chat heartbeat handles this;
do not install another cron job or duplicate watcher.

Read `work/reasoning-repair-v4/state.json` first. Do not start training until the
state is `ready`, `waiting_gpu`, or `waiting_disk`. Then use:

```sh
python3 experiments/reasoning_repair_v4/workflow.py \
  --project /root/workspace/train_exp/FACET-Terminal \
  --run /root/workspace/train_exp/FACET-Terminal/work/reasoning-repair-v4 \
  --launch
```

The launcher and worker use a live file lock to prevent duplicates. The worker
requires all eight GPUs to have no compute processes, less than 1 GiB used, and
less than 5% utilization in two checks 30 seconds apart. At least 40 GiB of disk
must be free before training. No other process is stopped. A running worker is
monitored through its state, logs, and markers; never launch a second one.

The worker performs an eight-step pilot, checks finite loss and JSON generation,
then trains a fresh model from the base for three epochs. Only model weights are
checkpointed, avoiding large optimizer snapshots on the space-constrained disk.
Failures preserve partial outputs and require inspection; they are not silently
overwritten or retried. Only child processes created by this worker are stopped.
Global Docker cleanup, retagging base images, killing other GPU jobs, and deleting
old experiments are prohibited.

## Evaluation and reporting

The three arms are base Qwen3.5-4B, the official FACET 4B checkpoint, and the new
model. All use Terminus-2, temperature 1.0, explicit thinking, context 32,768,
three attempts for each of 89 tasks, and an explicit 7,200-second agent timeout.
The all-model 32K context is a matched local protocol; the paper's base-model
context may differ. Keep that distinction in the final comparison.

Models are served one at a time on port 18040 with a dedicated process group.
No existing listener is replaced. Model metadata is explicitly passed to Harbor
to avoid its unknown-model context fallback; `llm_call_kwargs.extra_body` is used
for request parameters. The original task timeouts do not override the explicit
agent timeout. Verifier task limits remain frozen and infrastructure failures
remain visible. Automatic evaluation retries are disabled.

`score.py` always uses denominator 267. Missing verdicts do not shrink it.
Conflicting `result.json` and `reward.txt` values block the reported score;
a concrete reward file can fill a missing serialized reward. Infrastructure
failures require review and are never selectively retried based on reward.

Keep the heartbeat quiet while GPUs are busy or the run progresses normally.
Notify when training starts, a stage completes, an error occurs, or the user
must act. After the workflow completes, analyze the raw results and uncertainty,
report limitations, and stop the heartbeat. Do not call a launch a completed run.

## Checks

```sh
cd experiments/reasoning_repair_v4
python3 -m unittest test_workflow.py
```

These tests cover the data-loss regression, matched control, fixed denominator,
conflicting verdicts, and prevention of launch when another compute job exists.
Real label and Harbor checks run separately during CPU preparation.
