"""Fixed-denominator scoring. Missing and conflicting verdicts remain visible."""
import argparse
import collections
import json
from pathlib import Path

from prepare import dump


def summarize(job, task_names, attempts=3):
    rows = collections.defaultdict(list)
    conflicts = []
    for path in sorted(job.glob("*/result.json")):
        record = json.loads(path.read_text())
        task = record["task_name"].split("/")[-1]
        if task not in task_names:
            raise ValueError(f"Unexpected task {task}")
        reward = ((record.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        rp = path.parent / "verifier/reward.txt"
        file_reward = float(rp.read_text().strip()) if rp.exists() else None
        if reward is not None and file_reward is not None and reward != file_reward:
            conflicts.append(path.parent.name)
        # reward.txt rescues a timeout written after the result serializer; never
        # rescue a genuine disagreement between two concrete verdicts.
        chosen = reward if reward is not None else file_reward
        rows[task].append({"trial": path.parent.name, "reward": chosen,
                           "exception": (record.get("exception_info") or {}).get("exception_type")})
    if any(len(v) > attempts for v in rows.values()):
        raise ValueError("Extra attempts would change the fixed trial population")
    passes = sum(r["reward"] == 1 for rs in rows.values() for r in rs)
    completed = sum(r["reward"] is not None for rs in rows.values() for r in rs)
    infrastructure = [r for rs in rows.values() for r in rs
                      if r["exception"] and r["exception"] != "AgentTimeoutError"]
    denominator = len(task_names) * attempts
    result = {"passes": passes, "denominator": denominator, "percent": 100 * passes / denominator,
              "attempt_records": sum(map(len, rows.values())), "verdicts": completed,
              "missing_verdict_slots": denominator - completed, "conflicting_verdicts": conflicts,
              "infrastructure_failures": infrastructure,
              "per_task": {task: rows[task] for task in task_names},
              "complete": completed == denominator and not conflicts and not infrastructure}
    if conflicts:
        result["percent"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    names = json.loads((args.run / "benchmark_manifest.json").read_text())["tasks"]
    result = {name: summarize(args.run / "eval" / f"v4-{name}", names)
              for name in ["base", "official", "v4"]}
    dump(args.run / "scores.json", result)
    print(json.dumps({k: {x: y for x, y in v.items() if x != "per_task"} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
