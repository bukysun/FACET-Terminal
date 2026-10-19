"""Validate generated configs using the installed Harbor schema and agent route."""
import argparse
import json
from pathlib import Path

from harbor.models.job.config import JobConfig
from harbor.agents.terminus_2.terminus_2 import Terminus2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    for name in ["base", "official", "v4"]:
        config = JobConfig.model_validate_json((args.run / f"configs/eval_{name}.json").read_text())
        agent = config.agents[0]
        assert agent.override_timeout_sec == 7200
        assert agent.kwargs["temperature"] == 1.0
        assert agent.kwargs["model_info"]["max_input_tokens"] == 32768
        assert agent.kwargs["llm_call_kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert config.n_attempts == 3 and config.retry.max_retries == 0
    # Inspect the callable signature: top-level extra_body must never reappear.
    import inspect
    assert "llm_call_kwargs" in inspect.signature(Terminus2.__init__).parameters
    print(json.dumps({"harbor_configs_valid": True, "models": 3, "timeout": 7200, "temperature": 1.0}))


if __name__ == "__main__":
    main()
