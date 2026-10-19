"""Build current-turn SFT examples from the exact previously selected v2 tasks.

Run in facet-train2. No GPU, model weights, network calls, or source mutations.
Generated data and private paths belong in ignored work/, never in Git.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def target(reasoning, content):
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"


def valid_action(content):
    try:
        obj = json.loads(content)
    except (ValueError, TypeError):
        return False
    return (isinstance(obj, dict) and isinstance(obj.get("commands"), list)
            and isinstance(obj.get("analysis"), str) and isinstance(obj.get("plan"), str))


def make_example(messages, end, with_reasoning=True):
    """History contains content only; the one current target contains its reasoning."""
    result = []
    for i, message in enumerate(messages[:end + 1]):
        content = message["content"]
        if i == end:
            reasoning = (message.get("reasoning_content") or "") if with_reasoning else ""
            content = target(reasoning, content)
        result.append({"from": "human" if message["role"] == "user" else "gpt", "value": content})
    return {"conversations": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/models/Qwen3.5-4B"))
    parser.add_argument("--cutoff", type=int, default=32768)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    selected_path = args.project / "data/facet_sft_v2_1200.json"
    selected = [json.loads(line) for line in selected_path.open() if line.strip()]
    task_texts = {r["conversations"][0]["value"] for r in selected}
    if len(task_texts) != len(selected):
        raise ValueError("Selected task instructions are not unique")
    args.output.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    data_args = DataArguments(template="qwen3_5", mask_history=True, cutoff_len=args.cutoff)
    template = get_template_and_fix_tokenizer(tok, data_args)
    if template.thought_words != ("<think>\n", "\n</think>\n\n"):
        # Rendering is validated below too; record any change rather than assume it.
        print("Template thought markers:", repr(template.thought_words), flush=True)

    matches = []
    for path in sorted((args.project / "jobs/facet-rollout-v2").glob("task_*/result.json")):
        reward = path.parent / "verifier/reward.txt"
        if not reward.exists() or reward.read_text().strip() != "1":
            continue
        raw = json.loads(path.read_text())
        messages = ((raw.get("agent_result") or {}).get("metadata") or {}).get("all_messages") or []
        if not messages or "Task Description:\n" not in messages[0].get("content", ""):
            continue
        if messages[0]["content"].split("Task Description:\n", 1)[1] in task_texts:
            matches.append((path, messages, raw.get("task_checksum")))
    if len(matches) != len(selected):
        raise ValueError(f"Expected {len(selected)} unique source tasks, found {len(matches)}")
    if len({p.parent.name.split("__")[0] for p, _, _ in matches}) != len(matches):
        raise ValueError("Multiple trials for a selected source task")

    pilot_tasks = set(random.Random(42).sample([p.parent.name for p, _, _ in matches], min(64, len(matches))))
    counts = collections.Counter()
    lengths = []
    sources = []
    handles = {name: (args.output / f"{name}.jsonl").open("w") for name in ["train", "control", "pilot", "examples"]}
    try:
        for number, (path, messages, checksum) in enumerate(matches, 1):
            if len(messages) % 2 or any(m.get("role") != ("user" if i % 2 == 0 else "assistant")
                                       or not isinstance(m.get("content"), str) for i, m in enumerate(messages)):
                raise ValueError(f"Nonlinear or malformed source: {path}")
            sources.append({"trial": path.parent.name, "sha256": digest(path), "task_checksum": checksum})
            plain = [{"role": m["role"], "content": m["content"]} for m in messages]
            # Encode each history segment once. No history thoughts are present.
            # Each target is encoded separately, retaining its entire reasoning.
            parts = template._encode(tok, plain, "", None)
            prefix_length = 0
            for i in range(0, len(messages), 2):
                prefix_length += len(parts[i])
                end = i + 1
                message = messages[end]
                counts["source_turns"] += 1
                if not valid_action(message["content"]):
                    counts["invalid_target_json"] += 1
                    prefix_length += len(parts[end])
                    continue
                current = target(message.get("reasoning_content") or "", message["content"])
                if any(marker in current for marker in ("<image>", "<video>", "<audio>")):
                    counts["multimodal_literal_target_dropped"] += 1
                    prefix_length += len(parts[end])
                    continue
                pair = template.encode_multiturn(tok, [{"role": "user", "content": ""},
                                                       {"role": "assistant", "content": current}],
                                                system="", discarding_history_cot=True)[-1]
                length = prefix_length + len(pair[1])
                prefix_length += len(parts[end])
                if length > args.cutoff:
                    counts["over_cutoff_dropped_without_truncation"] += 1
                    continue
                example = make_example(messages, end)
                control = make_example(messages, end, with_reasoning=False)
                if counts["examples"] < 20:
                    full = [{"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
                            for c in example["conversations"]]
                    pairs = template.encode_multiturn(tok, full, system="", discarding_history_cot=True)
                    if sum(len(a) + len(b) for a, b in pairs) != length:
                        raise AssertionError("Cached length differs from actual LLaMA-Factory encoding")
                handles["train"].write(json.dumps(example, ensure_ascii=False) + "\n")
                handles["control"].write(json.dumps(control, ensure_ascii=False) + "\n")
                if path.parent.name in pilot_tasks:
                    handles["pilot"].write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts["pilot_examples"] += 1
                handles["examples"].write(json.dumps({"trial": path.parent.name, "assistant_turn": end // 2,
                    "tokens": length, "reasoning_chars": len(message.get("reasoning_content") or ""),
                    "target_sha256": hashlib.sha256(current.encode()).hexdigest()}) + "\n")
                lengths.append(length)
                counts["examples"] += 1
                counts["reasoning_examples"] += bool(message.get("reasoning_content"))
                counts["reasoning_chars_retained"] += len(message.get("reasoning_content") or "")
            if number % 100 == 0:
                print(f"Prepared {number}/{len(matches)} source tasks; {counts['examples']} targets", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    if not lengths:
        raise ValueError("No training examples")
    info = {name: {"file_name": f"{name}.jsonl", "formatting": "sharegpt", "columns": {"messages": "conversations"}}
            for name in ["train", "control", "pilot"]}
    dump(args.output / "dataset_info.json", info)
    versions = {p: importlib.metadata.version(p) for p in ["transformers", "torch", "llamafactory", "datasets"]}
    manifest = {"source_selection_sha256": digest(selected_path), "source_tasks": len(matches),
                "counts": dict(counts), "cutoff": args.cutoff, "max_tokens": max(lengths),
                "versions": versions, "sources": sources,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.project, text=True).strip(),
                "design": "Raw Harbor user prompt; per-turn prefix; history content only; complete current reasoning; last-turn loss only",
                "overlength_policy": "Drop the example, never truncate current target or silently shorten task history",
                "training_unit": "one assistant turn, not one complete trajectory; expanded sample count changes optimizer steps",
                "files": {p.name: digest(p) for p in args.output.glob("*.jsonl")}}
    dump(args.output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ["sources", "files"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
