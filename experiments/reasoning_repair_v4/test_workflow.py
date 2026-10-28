"""Critical regression checks; no training libraries or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare import make_example, valid_action
from score import summarize
from workflow import child_env, evaluation_command, free_disk_bytes, resources, state
from prepare import digest
from types import SimpleNamespace


class RegressionTests(unittest.TestCase):
    def test_resume_requires_review_and_preserves_completed_errors(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            job = run / "eval/v4-base"
            trial = job / "task__one"
            trial.mkdir(parents=True)
            (job / "config.json").write_text("{}")
            result = trial / "result.json"
            result.write_text('{"exception_info":{"exception_type":"CancelledError"}}')
            with self.assertRaises(RuntimeError):
                evaluation_command(run, "base")
            (run / "resume_base.json").write_text(json.dumps({
                "job_config_sha256": digest(job / "config.json"),
                "preserved_results": {"task__one/result.json": digest(result)}}))
            command = evaluation_command(run, "base")
            self.assertIn("resume", command)
            self.assertEqual(command[-1], "__FACET_KEEP_ALL_COMPLETED_RESULTS__")
            (job / "unfinished").mkdir()
            with self.assertRaises(RuntimeError):
                evaluation_command(run, "base")

    def test_system_disk_is_checked_after_project_migration(self):
        with patch("workflow.shutil.disk_usage", side_effect=[
            SimpleNamespace(free=1000), SimpleNamespace(free=5)
        ]):
            self.assertEqual(free_disk_bytes(Path("/data/project")), 5)

    def test_console_environment_precedes_system_path(self):
        env = child_env(bin_dir="/example/train/bin")
        self.assertEqual(env["PATH"].split(":")[0], "/example/train/bin")

    def test_each_action_retains_current_reasoning_only(self):
        messages = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "action one", "reasoning_content": "reason one"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "action two", "reasoning_content": "reason two"}]
        first = make_example(messages, 1)["conversations"]
        second = make_example(messages, 3)["conversations"]
        self.assertIn("reason one", first[-1]["value"])
        self.assertEqual(second[1]["value"], "action one")
        self.assertIn("reason two", second[-1]["value"])
        self.assertNotIn("reason one", json.dumps(second))
        control = make_example(messages, 3, False)["conversations"]
        self.assertEqual(control[:-1], second[:-1])
        self.assertNotIn("reason two", control[-1]["value"])

    def test_score_never_shrinks_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            job = Path(root)
            trial = job / "task_a__one"
            trial.mkdir()
            (trial / "result.json").write_text(json.dumps({"task_name": "x/task_a",
                "verifier_result": {"rewards": {"reward": 1}}}))
            result = summarize(job, ["task_a", "task_b"], attempts=3)
            self.assertEqual(result["denominator"], 6)
            self.assertEqual(result["missing_verdict_slots"], 5)
            self.assertAlmostEqual(result["percent"], 100 / 6)
            self.assertFalse(result["complete"])

    def test_conflicting_rewards_block_score(self):
        with tempfile.TemporaryDirectory() as root:
            job = Path(root)
            trial = job / "a"
            (trial / "verifier").mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps({"task_name": "task",
                "verifier_result": {"rewards": {"reward": 1}}}))
            (trial / "verifier/reward.txt").write_text("0")
            self.assertIsNone(summarize(job, ["task"], 1)["percent"])

    def test_foreign_compute_process_prevents_launch(self):
        output = "\n".join(f"{i}, 0, 0" for i in range(8))
        with patch("subprocess.check_output", side_effect=[output, "1234\n"]):
            self.assertFalse(resources()["all_eight_idle"])
        with patch("subprocess.check_output", side_effect=[output, ""]):
            self.assertTrue(resources()["all_eight_idle"])

    def test_successful_state_does_not_keep_old_failure(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            state(run, "preparation_failed", error="old failure")
            state(run, "ready")
            self.assertNotIn("error", json.loads((run / "state.json").read_text()))


if __name__ == "__main__":
    unittest.main()
