"""Resource-gated, resumable FACET experiment; only manages its own child processes."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from prepare import digest, dump

ENVS = Path("/root/miniconda3/envs")
SCRIPT_DIR = Path(__file__).resolve().parent


def state(run, stage, **fields):
    path = run / "state.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    value = previous | fields | {"stage": stage, "updated_at_epoch": time.time()}
    if "error" not in fields:
        value.pop("error", None)
    temporary = path.with_suffix(".tmp")
    dump(temporary, value)
    temporary.replace(path)
    print(json.dumps(value), flush=True)


def resources():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    gpus = [[int(x.strip()) for x in line.split(",")] for line in output.splitlines()]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    free = len(gpus) == 8 and not processes and all(memory < 1024 and util < 5 for _, memory, util in gpus)
    return {"all_eight_idle": free, "gpus": gpus, "compute_pids": processes.splitlines() if processes else []}


def verify_inputs(run):
    for name, sha in json.loads((run / "input_hashes.json").read_text()).items():
        assert digest(run / name) == sha, f"Frozen input changed: {name}"
    validation = json.loads((run / "data/validation.json").read_text())
    assert validation["passed"]
    manifest = json.loads((run / "data/manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert digest(run / "data" / name) == sha, f"Data changed: {name}"
    benchmark = json.loads((run / "benchmark_manifest.json").read_text())
    for name, sha in benchmark["files"].items():
        assert digest(run / "terminal-bench-2-1" / name) == sha, f"Benchmark changed: {name}"
    for name, sha in json.loads((run / "runtime_inputs.json").read_text()).items():
        assert digest(name) == sha, f"Model or installed runtime changed: {name}"


def child_env(cuda_home=None):
    env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OPENAI_API_KEY="dummy-local-key",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost", DOCKER_BUILDKIT="1")
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
    return env


def stop_own(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_child(command, project, log, env):
    with log.open("a") as output:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if shutil.disk_usage(project).free < 10 * (1 << 30):
                    raise RuntimeError("Less than 10 GiB disk left; stopping our own child only")
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"Child exited {process.returncode}; inspect {log}")
        finally:
            stop_own(process)


def get_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return json.load(response)


def start_server(model, name, project, run):
    assert resources()["all_eight_idle"], "GPU resources changed before starting our server"
    # An existing listener is never killed or replaced.
    with socket.socket() as test:
        assert test.connect_ex(("127.0.0.1", 18040)) != 0, "Port 18040 is already occupied"
    log = (run / f"serve-{name}.log").open("a")
    command = [str(ENVS / "vllm-serve/bin/vllm"), "serve", str(model), "--host", "127.0.0.1",
        "--port", "18040", "--served-model-name", f"facet-v4-{name}", "--max-model-len", "32768",
        "--reasoning-parser", "qwen3", "--data-parallel-size", "8", "--gpu-memory-utilization", "0.85",
        "--generation-config", "vllm", "--seed", "42"]
    process = subprocess.Popen(command, cwd=project, env=child_env(str(ENVS / "vllm-serve")),
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    try:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError(f"vLLM failed: serve-{name}.log")
            try:
                models = get_json("http://127.0.0.1:18040/v1/models")
                if f"facet-v4-{name}" in [x["id"] for x in models["data"]]:
                    return process
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError("vLLM readiness timed out")
    except BaseException:
        stop_own(process)
        raise


def pilot_sanity(project, run):
    process = start_server(run / "models/pilot", "pilot", project, run)
    try:
        record = json.loads(next((run / "data/pilot.jsonl").open()))
        prompt = record["conversations"][0]["value"].split("Task Description:\n", 1)[0]
        prompt += "Task Description:\nThis is a JSON formatting check. Return commands=[] and task_complete=true. No shell commands are needed."
        body = {"model": "facet-v4-pilot", "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0, "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True}}
        req = urllib.request.Request("http://127.0.0.1:18040/v1/chat/completions",
              data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=900) as response:
            answer = json.load(response)
        dump(run / "pilot_sanity_response.json", answer)
        message = answer["choices"][0]["message"]
        value = json.loads(message["content"])
        assert isinstance(value.get("commands"), list) and "analysis" in value and "plan" in value
        assert answer["choices"][0]["finish_reason"] != "length", "Sanity generation truncated"
    finally:
        stop_own(process)
        time.sleep(10)


def worker(project, run):
    with (run / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state(run, "checking_resources", worker_pid=os.getpid())
        check = resources()
        if not check["all_eight_idle"]:
            state(run, "waiting_gpu", worker_pid=None, resources=check)
            return
        time.sleep(30)
        check = resources()
        if not check["all_eight_idle"]:
            state(run, "waiting_gpu", worker_pid=None, resources=check)
            return
        if shutil.disk_usage(project).free < 40 * (1 << 30):
            state(run, "waiting_disk", worker_pid=None, error="Need at least 40 GiB free before training")
            return
        try:
            verify_inputs(run)
            check = resources()
            if not check["all_eight_idle"]:
                state(run, "waiting_gpu", worker_pid=None, resources=check)
                return
            for phase in ["pilot", "train"]:
                done = run / f"{phase}.done.json"
                if done.exists():
                    continue
                # Partial training outputs are retained for inspection, never silently overwritten.
                model_dir = run / "models" / ("pilot" if phase == "pilot" else "v4")
                if model_dir.exists() and any(model_dir.iterdir()):
                    raise RuntimeError(f"Partial output exists: {model_dir}; inspect before retrying")
                state(run, "pilot_training" if phase == "pilot" else "full_training")
                assert resources()["all_eight_idle"], "GPU resources changed before starting training"
                run_child([str(ENVS / "facet-train2/bin/llamafactory-cli"), "train",
                           str(run / f"configs/{phase}.json")], project, run / f"{phase}.log",
                          child_env(str(ENVS / "swift")))
                result = json.loads((model_dir / "train_results.json").read_text())
                assert math.isfinite(result["train_loss"]), "Nonfinite training loss"
                if phase == "pilot":
                    state(run, "pilot_sanity")
                    pilot_sanity(project, run)
                dump(done, result)
            for name, model in [("base", Path("/root/models/Qwen3.5-4B")),
                                ("official", Path("/root/models/FACET-Terminal-Qwen3.5-4B")),
                                ("v4", run / "models/v4")]:
                done = run / f"eval_{name}.done.json"
                if done.exists():
                    continue
                job = run / "eval" / f"v4-{name}"
                if job.exists():
                    raise RuntimeError(f"Partial evaluation exists: {job}; inspect before resuming")
                assert shutil.disk_usage(project).free > 20 * (1 << 30), "Need 20 GiB for evaluation"
                state(run, f"evaluating_{name}")
                process = start_server(model, name, project, run)
                try:
                    run_child([str(ENVS / "facet-rollout/bin/harbor"), "run", "-c",
                               str(run / f"configs/eval_{name}.json")], project, run / f"eval_{name}.log", child_env())
                finally:
                    stop_own(process)
                    time.sleep(10)
                dump(done, {"job": str(job)})
            run_child([sys.executable, str(SCRIPT_DIR / "score.py"), "--run", str(run)],
                      project, run / "score.log", child_env())
            scores = json.loads((run / "scores.json").read_text())
            state(run, "complete" if all(s["complete"] for s in scores.values()) else "needs_verdict_review",
                  worker_pid=None, scores={k: {x: y for x, y in v.items() if x != "per_task"} for k, v in scores.items()})
        except Exception as exc:
            state(run, "failed", worker_pid=None, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.project.resolve(), args.run.resolve())
        return
    # A live flock is authoritative; stale PIDs cannot cause duplicate jobs.
    with (args.run / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Workflow already running", (args.run / "state.json").read_text())
            return
        current = json.loads((args.run / "state.json").read_text())
        if current["stage"] not in ["ready", "waiting_gpu", "waiting_disk"]:
            print(json.dumps(current))
            return
        check = resources()
        print(json.dumps(check))
        if not check["all_eight_idle"]:
            state(args.run, "waiting_gpu", resources=check)
            return
    if args.launch:
        with (args.run / "workflow.log").open("a") as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--project", str(args.project.resolve()), "--run", str(args.run.resolve()), "--worker"],
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print("Spawned guarded workflow", process.pid)


if __name__ == "__main__":
    main()
