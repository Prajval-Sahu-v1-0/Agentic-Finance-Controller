"""
launcher.py — Native desktop entry point for the packaged .exe
==================================================================

What "run the .exe and it's ready to use" actually means here: start the
API server in the background, wait for it to come up, then open it in a
dedicated native application window (not a browser tab) — via pywebview,
which wraps the OS's own web renderer (Edge WebView2 on Windows, already
preinstalled on Windows 10/11 — no extra runtime to ship). Closing that
window stops the app.

This is the file PyInstaller's --onefile build points at — NOT
`uvicorn src.api:app` directly, since uvicorn's CLI reload/multiprocessing
machinery doesn't play well with a frozen single-file bundle, and NOT
"open the default browser," since the brief was a dedicated application
window.

Synthetic data: development convenience only, per explicit instruction —
never in the shipped .exe. Gated on `src.paths.is_frozen()`, which already
exists for path resolution (bundled vs. source), so no separate flag is
needed: running from source (`python -m src.launcher`) auto-generates a
sample dataset so there's something to click through immediately; the
actual packaged .exe never does. A user's real financial data goes in via
POST /upload (see src/ingest.py) — the point of an offline-first, one-file
app is that nothing about the user's data leaves their machine, and that
starts with not shipping fake data alongside it.

Usage
-----
    python -m src.launcher              # run from source (dev mode)
    recon-agent.exe                     # the packaged build (production mode)
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request

import uvicorn

from src.paths import DATA_DIR, is_frozen

HOST = "127.0.0.1"
PORT = 8756  # arbitrary, chosen to avoid colliding with common local dev ports
WINDOW_TITLE = "recon-agent — AI Finance Controller"


def _ensure_dev_sample_dataset() -> None:
    """Dev-only convenience: auto-generate the synthetic dataset if
    missing, so a fresh checkout has something to explore immediately.
    NEVER runs in the packaged .exe — see module docstring."""
    if is_frozen():
        return
    gw_path = DATA_DIR / "gateway_records.json"
    if gw_path.exists():
        return
    print("[recon-agent] (dev mode) No sample dataset found — generating one for local exploration...")
    from src.generator import generate_dataset
    generate_dataset(seed=42, output_dir=DATA_DIR, save=True)


def _wait_until_ready(url: str, timeout_seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def _run_server() -> None:
    from src.api import app
    logging.getLogger("uvicorn.access").disabled = True  # keep the console quiet in the packaged app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def main() -> None:
    _ensure_dev_sample_dataset()

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    base_url = f"http://{HOST}:{PORT}"
    print(f"[recon-agent] Starting on {base_url} ...")
    if not _wait_until_ready(f"{base_url}/health"):
        print(
            f"[recon-agent] The server did not become ready in time. "
            f"You can still try opening {base_url}/ manually, or check "
            f"for a port conflict on {PORT}."
        )

    try:
        import webview
    except ImportError:
        print(
            "[recon-agent] pywebview is not installed — falling back to your "
            f"default browser at {base_url}/. Install pywebview for a "
            "dedicated application window instead (pip install pywebview)."
        )
        import webbrowser
        webbrowser.open(base_url)
        print("[recon-agent] Press Ctrl+C to stop the app.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    webview.create_window(WINDOW_TITLE, base_url, width=1200, height=800, min_size=(800, 600))
    webview.start()
    # webview.start() blocks until the window is closed; the server thread
    # is a daemon, so the process exits cleanly once we return here.


if __name__ == "__main__":
    main()
