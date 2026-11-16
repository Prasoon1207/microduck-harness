#!/usr/bin/env python3
"""Single-command Uvicorn launcher for the interactive MicroDuck Harness UI."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import webbrowser

import uvicorn

from src.ui_server import EventBroker, HarnessChatSession, LiveDuckEngine, create_app


def _pick_port(host: str, preferred: int) -> int:
    probe_host = "127.0.0.1" if host in {"", "0.0.0.0", "::", "localhost"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if s.connect_ex((probe_host, preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((probe_host, 0))
        return int(s.getsockname()[1])


def _open_standalone_browser(url: str) -> None:
    """Open URL directly in standalone Google Chrome on macOS (avoiding IDE embedded webviews)."""
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["open", "-a", "Google Chrome", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the interactive MicroDuck Harness UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    parser.add_argument(
        "--provider",
        default="auto",
        help="Initial LLM provider ('auto', 'gemini', 'openai', 'anthropic', 'openrouter', 'deepseek', 'custom', or 'demo').",
    )
    parser.add_argument("--model", default=None, help="Override initial model name (e.g. gemini-3.5-flash, gemini-pro-latest).")
    parser.add_argument("--base-url", default=None, help="Optional custom OpenAI-compatible base URL.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    parser.add_argument(
        "--coordinate-mode",
        default="raw_documented",
        choices=["raw_documented", "raw_undocumented", "normalized_symmetric"],
    )
    parser.add_argument("--max-chunks", type=int, default=10, help="Max closed-loop <action_chunk> turns per prompt.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open Chrome on startup.")
    args = parser.parse_args()

    port = _pick_port(args.host, args.port)
    display_host = "127.0.0.1" if args.host in {"", "0.0.0.0", "::", "localhost", "127.0.0.1"} else args.host
    url = f"http://{display_host}:{port}"

    broker = EventBroker()
    engine = LiveDuckEngine(
        broker=broker,
        coordinate_mode=args.coordinate_mode,
        enable_renderer=True,
        realtime_chunks=True,
    )
    session = HarnessChatSession(
        engine=engine,
        broker=broker,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        coordinate_mode=args.coordinate_mode,
        max_chunks_per_prompt=args.max_chunks,
    )
    app = create_app(engine=engine, session=session, broker=broker)

    sockets = None
    if args.host in {"127.0.0.1", "localhost", "::", "0.0.0.0", ""} and socket.has_ipv6:
        dual_sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        dual_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        dual_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        dual_sock.bind(("::", port))
        dual_sock.setblocking(False)
        sockets = [dual_sock]

    print(
        f"\nMicroDuck Harness UI (Uvicorn): {url} (and http://localhost:{port})\n"
        f"  physics  : 50 Hz MuJoCo (14-DOF XL330)\n"
        f"  provider : {session.provider} ({session.model})\n"
        f"  Ctrl+C to stop.\n",
        flush=True,
    )

    if not args.no_browser:
        threading.Timer(0.45, lambda: _open_standalone_browser(url)).start()

    config = uvicorn.Config(
        app,
        host=args.host,
        port=port,
        log_level="warning",
        access_log=False,
        ws="none",
    )
    server = uvicorn.Server(config)

    try:
        server.run(sockets=sockets)
    except KeyboardInterrupt:
        pass
    finally:
        if sockets:
            for s in sockets:
                try:
                    s.close()
                except Exception:
                    pass
        engine.close()


if __name__ == "__main__":
    main()
