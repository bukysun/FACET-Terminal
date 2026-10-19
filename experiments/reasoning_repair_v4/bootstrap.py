"""Prepare and validate the new run on CPU. Safe to leave running after SSH exits."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from workflow import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume a failed preparation with completed data manifest")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "bootstrap.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resuming = (args.run / "state.json").exists()
        if resuming:
            previous = json.loads((args.run / "state.json").read_text())
            if not args.resume or previous["stage"] != "preparation_failed" or not (args.run / "data/manifest.json").exists():
                raise RuntimeError("Existing preparation cannot be safely resumed; inspect it first")
        script = Path(__file__).resolve().parent
        train_python = "/root/miniconda3/envs/facet-train2/bin/python"
        state(args.run, "preparing_data")
        try:
            commands = [
                [train_python, str(script / "prepare.py"), "--project", str(args.project), "--output", str(args.run / "data")],
                [train_python, str(script / "validate_labels.py"), "--data", str(args.run / "data")],
                [sys.executable, str(script / "configure.py"), "--project", str(args.project), "--run", str(args.run)],
                ["/root/miniconda3/envs/facet-rollout/bin/python", str(script / "validate_configs.py"), "--run", str(args.run)],
            ]
            if resuming:
                commands = commands[1:]
                if (args.run / "input_hashes.json").exists():
                    commands = [command for command in commands if not command[1].endswith("configure.py")]
            for command in commands:
                subprocess.run(command, cwd=args.project, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            state(args.run, "ready", next_action="Wait for all eight GPUs to be idle, then workflow.py --launch")
        except Exception as exc:
            state(args.run, "preparation_failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
