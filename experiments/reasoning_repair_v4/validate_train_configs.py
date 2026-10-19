"""Load configs through the installed LF CLI reader without initializing CUDA."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

from prepare import digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from llamafactory.hparams.parser import _TRAIN_ARGS, read_args
    known = {field.name for cls in _TRAIN_ARGS for field in fields(cls)}
    checked = {}
    original_argv = sys.argv[:]
    try:
        for name in ["train", "pilot", "control_train"]:
            path = args.run / f"configs/{name}.yaml"
            sys.argv = [__file__, str(path.resolve())]
            value = read_args()
            assert isinstance(value, dict), "LF CLI did not load a config dictionary"
            assert not set(value) - known, f"Unknown training keys: {set(value) - known}"
            assert isinstance(value["learning_rate"], float) and value["learning_rate"] == 1e-5
            assert value["mask_history"] is True and value["enable_thinking"] is True
            assert value["cutoff_len"] == 32768 and value["gradient_accumulation_steps"] == 8
            checked[str(path.relative_to(args.run))] = digest(path)
    finally:
        sys.argv = original_argv
    report = {"passed": True, "configs": checked, "checks": ["installed LF CLI loader", "known dataclass fields", "typed hyperparameters"]}
    dump(args.run / "train_config_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
