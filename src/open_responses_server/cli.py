"""Command line interface: serve an agent over the Open Responses API.

Usage::

    open-responses-server serve my_module:agent --port 8080
    open-responses-server serve path/to/agent.py:agent --api-key secret

The target may resolve to either an :class:`AgentAdapter` instance or a
Google ADK ``BaseAgent`` (which is wrapped in an ``ADKAdapter``
automatically).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from typing import Any

from .adapter import AgentAdapter


def load_target(target: str) -> Any:
    """Load ``module:attribute`` or ``path/to/file.py:attribute``."""
    module_ref, sep, attribute = target.partition(":")
    if not sep or not module_ref or not attribute:
        raise SystemExit(
            f"Invalid target '{target}'. Expected 'module:attribute' or "
            "'path/to/file.py:attribute'."
        )

    if module_ref.endswith(".py") or os.path.sep in module_ref:
        path = os.path.abspath(module_ref)
        spec = importlib.util.spec_from_file_location("_or_target_module", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Cannot load python file '{module_ref}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(module_ref)

    try:
        obj: Any = module
        for name in attribute.split("."):
            obj = getattr(obj, name)
    except AttributeError:
        raise SystemExit(f"Module '{module_ref}' has no attribute '{attribute}'.")
    return obj


def resolve_adapter(obj: Any, *, model_name: str | None = None) -> AgentAdapter:
    if isinstance(obj, AgentAdapter):
        return obj
    try:
        from google.adk.agents import BaseAgent
    except ImportError:
        BaseAgent = None  # type: ignore[assignment]
    if BaseAgent is not None and isinstance(obj, BaseAgent):
        from .adapters.adk import ADKAdapter

        return ADKAdapter(obj, model_name=model_name)
    raise SystemExit(
        f"Target of type {type(obj).__name__} is neither an AgentAdapter nor a "
        "google.adk BaseAgent. Install the framework extra (e.g. "
        "'open-responses-server[adk]') or point at an AgentAdapter instance."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-responses-server",
        description="Serve agent frameworks over the Open Responses API.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve an agent over HTTP.")
    serve.add_argument(
        "target",
        help="Agent or adapter to serve, as 'module:attribute' or "
        "'path/to/file.py:attribute'.",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--api-key",
        default=os.environ.get("OPEN_RESPONSES_API_KEY"),
        help="Require 'Authorization: Bearer <key>' on all requests "
        "(env: OPEN_RESPONSES_API_KEY).",
    )
    serve.add_argument(
        "--model-name",
        default=None,
        help="Model name advertised in responses when requests omit 'model'.",
    )
    serve.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from .server import create_app

        adapter = resolve_adapter(load_target(args.target), model_name=args.model_name)
        if args.model_name:
            adapter.default_model = args.model_name
        app = create_app(adapter, api_key=args.api_key)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
