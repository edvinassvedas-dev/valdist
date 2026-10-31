from __future__ import annotations

import re
import tomllib

import pytest
import yaml
from conftest import PROJECT_ROOT

WORKFLOW = yaml.safe_load((PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
STEPS = [step for job in WORKFLOW["jobs"].values() for step in job["steps"]]
ACTIONS = [step["uses"] for step in STEPS if "uses" in step]
COMMANDS = [step["run"] for step in STEPS if "run" in step]


@pytest.mark.parametrize("action", ACTIONS)
def test_an_action_is_pinned_to_a_full_commit(action) -> None:
    assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action)


def test_every_job_names_its_runner_image() -> None:
    assert not any(j["runs-on"].endswith("-latest") for j in WORKFLOW["jobs"].values())


def test_ci_installs_from_the_lock() -> None:
    assert any("uv sync --locked" in c for c in COMMANDS)
    assert not any(re.search(r"\bpip install\b", c) for c in COMMANDS)


def test_ci_audits_the_lock() -> None:
    assert any("uv audit" in c and "--locked" in c for c in COMMANDS)


def test_ci_runs_the_oldest_python_and_one_past_the_lock_fork() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    oldest = project["project"]["requires-python"].removeprefix(">=")
    versions = WORKFLOW["jobs"]["ci"]["strategy"]["matrix"]["python-version"]
    assert oldest in versions
    assert any(int(v.split(".")[1]) >= 12 for v in versions)
