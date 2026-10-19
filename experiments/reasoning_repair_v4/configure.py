"""Freeze local experiment inputs and generate explicit train/evaluation configs."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from prepare import digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    project = args.project.resolve()
    data = run / "data"
    validation = json.loads((data / "validation.json").read_text())
    assert validation["passed"] and validation["manifest_sha256"] == digest(data / "manifest.json")
    configs = run / "configs"
    configs.mkdir(exist_ok=False)
    tasks = run / "terminal-bench-2-1"
    shutil.copytree(project / "terminal-bench-2-1", tasks)
    task_names = sorted(p.name for p in tasks.iterdir() if (p / "task.toml").is_file())
    assert len(task_names) == 89
    dump(run / "benchmark_manifest.json", {"tasks": task_names,
        "files": {str(p.relative_to(tasks)): digest(p) for p in sorted(tasks.rglob("*")) if p.is_file()}})
    zero = {
        "train_batch_size": "auto", "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto", "gradient_clipping": 1.0,
        "bf16": {"enabled": "auto"}, "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True,
            "reduce_bucket_size": "auto", "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto", "stage3_gather_16bit_weights_on_model_save": True}}
    dump(configs / "zero3.json", zero)
    train = {"stage": "sft", "do_train": True, "finetuning_type": "full",
        "model_name_or_path": "/root/models/Qwen3.5-4B", "trust_remote_code": True,
        "dataset": "train", "dataset_dir": str(data), "template": "qwen3_5",
        "mask_history": True, "enable_thinking": True, "preserve_thinking": False,
        "cutoff_len": 32768, "overwrite_cache": True, "preprocessing_num_workers": 8,
        "num_train_epochs": 3.0, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1,
        "bf16": True, "deepspeed": str(configs / "zero3.json"), "gradient_checkpointing": True,
        "enable_liger_kernel": True, "seed": 42, "data_seed": 42,
        "output_dir": str(run / "models/v4"), "logging_steps": 1,
        "save_strategy": "epoch", "save_only_model": True, "save_total_limit": 1,
        "report_to": "none", "plot_loss": True}
    dump(configs / "train.json", train)
    pilot = copy.deepcopy(train)
    pilot.update(dataset="pilot", max_steps=8, num_train_epochs=1.0,
                 output_dir=str(run / "models/pilot"), save_strategy="no", plot_loss=False)
    dump(configs / "pilot.json", pilot)
    control = copy.deepcopy(train)
    control.update(dataset="control", output_dir=str(run / "models/control"))
    dump(configs / "control_train.json", control)
    environment = json.loads((project / "jobs/facet-sft-v3-tb21-k3/config.json").read_text())["environment"]
    for name in ["base", "official", "v4"]:
        context = 32768
        agent = {"name": "terminus-2", "model_name": f"openai/facet-v4-{name}",
            "override_timeout_sec": 7200.0,
            "kwargs": {"api_base": "http://127.0.0.1:18040/v1", "temperature": 1.0,
                "record_terminal_session": False, "store_all_messages": True,
                "interleaved_thinking": False,
                "model_info": {"max_tokens": context, "max_input_tokens": context,
                               "max_output_tokens": context,
                               "input_cost_per_token": 0, "output_cost_per_token": 0},
                "llm_call_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}}
        job = {"job_name": f"v4-{name}", "jobs_dir": str(run / "eval"), "n_attempts": 3,
            "n_concurrent_trials": 8, "retry": {"max_retries": 0}, "environment": environment,
            "agents": [agent], "datasets": [{"path": str(tasks)}]}
        dump(configs / f"eval_{name}.json", job)
        smoke = copy.deepcopy(job)
        smoke.update(job_name=f"smoke-{name}", n_attempts=1, n_concurrent_trials=1)
        # Fixed names chosen before observing the repaired model's results.
        smoke["datasets"][0]["task_names"] = ["fix-git"]
        dump(configs / f"smoke_{name}.json", smoke)
    dump(run / "protocol.json", {"training_unit": "current assistant turn; not whole trajectory",
        "source": "same selected 1191 v2 tasks, expanded into per-turn examples",
        "train": {"epochs": 3, "effective_batch_turns": 64, "gpus": 8,
                  "note": "More optimizer steps than original 57-step trajectory training; not a single-variable causal ablation"},
        "evaluation": {"purpose": "matched local comparison, not exact reproduction of paper base context",
            "models": ["base", "official", "v4"], "context_all_models": 32768,
            "agent_timeout_sec": 7200, "attempts_per_task": 3, "temperature": 1.0,
            "denominator": 267, "infrastructure_failures": "retain and report; no reward-dependent retries"},
        "control": "Matched no-reasoning data and config are prepared; not automatically trained"})
    runtime_files = []
    packages = Path("/root/miniconda3/envs")
    for env, relative in [
        ("facet-train2", "lib/python3.11/site-packages/llamafactory/data/template.py"),
        ("facet-train2", "lib/python3.11/site-packages/llamafactory/data/processor/supervised.py"),
        ("facet-rollout", "lib/python3.12/site-packages/harbor/agents/terminus_2/terminus_2.py"),
        ("facet-rollout", "lib/python3.12/site-packages/harbor/llms/lite_llm.py"),
        ("facet-rollout", "lib/python3.12/site-packages/sitecustomize.py")]:
        path = packages / env / relative
        if path.exists():
            runtime_files.append(path)
    for model in [Path("/root/models/Qwen3.5-4B"), Path("/root/models/FACET-Terminal-Qwen3.5-4B")]:
        runtime_files.extend(model.glob("*.safetensors"))
        runtime_files.extend(model.glob("*.json"))
        runtime_files.extend(model.glob("*.jinja"))
    print("Hashing model and runtime inputs before the run", flush=True)
    dump(run / "runtime_inputs.json", {str(p): digest(p) for p in sorted(runtime_files)})
    dump(run / "input_hashes.json", {str(p.relative_to(run)): digest(p)
        for p in sorted(configs.glob("*.json"))} | {"protocol.json": digest(run / "protocol.json"),
            "benchmark_manifest.json": digest(run / "benchmark_manifest.json"),
            "data/manifest.json": digest(data / "manifest.json"),
            "data/validation.json": digest(data / "validation.json"),
            "runtime_inputs.json": digest(run / "runtime_inputs.json")})
    print("Created frozen configuration in", configs)


if __name__ == "__main__":
    main()
