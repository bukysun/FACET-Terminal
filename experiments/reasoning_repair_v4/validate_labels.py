"""Exercise the installed training processor, not an imitation of its labels."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import random

from prepare import digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="/root/models/Qwen3.5-4B")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data_args = DataArguments(template="qwen3_5", mask_history=True, enable_thinking=True, cutoff_len=32768)
    template = get_template_and_fix_tokenizer(tok, data_args)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tok, processor=None, data_args=data_args)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert digest(args.data / name) == sha, f"Changed dataset: {name}"
    n = 0
    for name in ["train", "control"]:
        # Deterministic reservoir sample spanning the entire dataset.
        sample = []
        rng = random.Random(42)
        for i, line in enumerate((args.data / f"{name}.jsonl").open()):
            if i < args.samples:
                sample.append(json.loads(line))
            else:
                j = rng.randrange(i + 1)
                if j < args.samples:
                    sample[j] = json.loads(line)
        for record in sample:
            messages = [{"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
                        for c in record["conversations"]]
            assert all("<think>" not in m["content"] for m in messages[1:-1:2])
            ids, labels = processor._encode_data_example(messages[:-1], messages[-1:], "", "", [], [], [])
            pairs = template.encode_multiturn(tok, messages, system="", tools="", discarding_history_cot=True)
            expected = pairs[-1][1]
            assert [v for v in labels if v != -100] == expected, "Labels include history or truncate current target"
            assert len(ids) == sum(len(a) + len(b) for a, b in pairs) <= 32768
            assert all(x == -100 for x in labels[:-len(expected)])
            decoded = tok.decode(expected, skip_special_tokens=False)
            assert decoded.startswith(messages[-1]["content"]), "Current reasoning differs from stored target"
            n += 1
    report = {"passed": True, "processor": dataclasses.asdict(data_args)["template"],
              "checked_examples": n, "manifest_sha256": digest(args.data / "manifest.json"),
              "checks": ["current target token-exact", "history labels all ignored", "no target truncation",
                         "no history reasoning", "all dataset file hashes unchanged"]}
    dump(args.data / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
