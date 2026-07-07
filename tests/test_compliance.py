"""Runs the official Open Responses compliance suite against this server.

Uses the CLI runner from https://github.com/openresponses/openresponses
(``bin/compliance-test.ts``) — the same suite as the web tester at
https://www.openresponses.org/compliance — pointed at a local server backed
by a deterministic ADK agent (see ``compliance_agent.py``).

Requirements: ``bun`` on PATH and network access to clone the spec repo on
first run (cached under ``.compliance/``). Skips itself otherwise.

Pin a different spec revision with ``OPENRESPONSES_SPEC_REF``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from open_responses_server.adapters.adk import ADKAdapter
from open_responses_server.server import create_app

from compliance_agent import create_agent

pytestmark = pytest.mark.compliance

SPEC_REPO_URL = os.environ.get(
    "OPENRESPONSES_SPEC_REPO", "https://github.com/openresponses/openresponses"
)
SPEC_REF = os.environ.get(
    "OPENRESPONSES_SPEC_REF", "cd31bc2060a27ee87a05ec97f49c84027eb6c3ba"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / ".compliance" / "openresponses"

# All HTTP-transport tests from the suite. WebSocket transport is not
# implemented by this server (yet), so those tests are excluded.
HTTP_TESTS = [
    "basic-response",
    "assistant-phase",
    "response-output-phase-schema",
    "streaming-response",
    "system-prompt",
    "tool-calling",
    "image-input",
    "multi-turn",
    "compact-response",
    "compact-missing-model",
]

API_KEY = "compliance-test-key"


@pytest.fixture(scope="session")
def bun() -> str:
    path = shutil.which("bun")
    if path is None:
        pytest.skip("bun is not installed; skipping compliance suite")
    return path


@pytest.fixture(scope="session")
def spec_repo(bun: str) -> Path:
    marker = CACHE_DIR / "bin" / "compliance-test.ts"
    head = CACHE_DIR / ".compliance-ref"
    if not marker.exists() or (head.exists() and head.read_text().strip() != SPEC_REF):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "init", "-q", str(CACHE_DIR)], check=True, timeout=60
            )
            subprocess.run(
                ["git", "fetch", "-q", "--depth", "1", SPEC_REPO_URL, SPEC_REF],
                cwd=CACHE_DIR,
                check=True,
                timeout=300,
            )
            subprocess.run(
                ["git", "checkout", "-q", "FETCH_HEAD"],
                cwd=CACHE_DIR,
                check=True,
                timeout=60,
            )
            _install_runtime_deps(bun)
            head.write_text(SPEC_REF)
        except (subprocess.SubprocessError, OSError) as exc:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            pytest.skip(f"could not fetch the openresponses spec repo: {exc}")
    return CACHE_DIR


def _install_runtime_deps(bun: str) -> None:
    """Install only the CLI's runtime dependencies (zod), not the full site.

    The spec repo's package.json includes the whole docs site (astro, react,
    ...); the compliance CLI only needs zod at the version pinned in the
    repo's package.json. Installing into a scratch package and moving the
    resulting node_modules into the cache keeps this fast and offline-cached.
    """
    package = json.loads((CACHE_DIR / "package.json").read_text())
    zod_version = package["dependencies"]["zod"]
    scratch = CACHE_DIR / ".deps"
    scratch.mkdir(exist_ok=True)
    (scratch / "package.json").write_text(
        json.dumps({"name": "compliance-deps", "dependencies": {"zod": zod_version}})
    )
    subprocess.run(
        [bun, "install", "--no-progress"],
        cwd=scratch,
        check=True,
        timeout=300,
        capture_output=True,
    )
    target = CACHE_DIR / "node_modules"
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(scratch / "node_modules", target)


@pytest.fixture(scope="session")
def server_url():
    adapter = ADKAdapter(create_agent(), app_name="compliance")
    app = create_app(adapter, api_key=API_KEY)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn server failed to start")
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=10)


def test_official_compliance_suite(bun: str, spec_repo: Path, server_url: str):
    result = subprocess.run(
        [
            bun,
            "run",
            "bin/compliance-test.ts",
            "--base-url",
            server_url,
            "--api-key",
            API_KEY,
            "--model",
            "compliance-model",
            "--json",
            "--filter",
            ",".join(HTTP_TESTS),
        ],
        cwd=spec_repo,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.stdout, f"compliance CLI produced no output: {result.stderr}"
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        pytest.fail(
            "compliance CLI did not emit JSON.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    failures = [r for r in report["results"] if r["status"] == "failed"]
    details = "\n\n".join(
        f"{r['name']} ({r['id']}):\n" + "\n".join(f"  - {e}" for e in r.get("errors") or [])
        for r in failures
    )
    assert not failures, f"compliance failures:\n{details}"
    assert report["summary"]["passed"] == len(HTTP_TESTS)
