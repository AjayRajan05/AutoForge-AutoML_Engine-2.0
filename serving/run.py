"""Uvicorn entrypoint for AutoForge serving API."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoForge REST serving API")
    parser.add_argument("--host", default=os.environ.get("AUTOFORGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUTOFORGE_PORT", "8000")))
    parser.add_argument("--model", default=os.environ.get("AUTOFORGE_MODEL_PATH"))
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install serving extras: pip install 'autoforge[serve]'") from exc

    from serving.app import create_app

    app = create_app(model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
