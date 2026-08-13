"""
=============================================================================
LAUNCHGUARD AI — LAUNCHER
=============================================================================

Double-click START.bat (Windows) or run `python launch.py`.

Does everything a teammate would otherwise have to do by hand:

  1. finds a working Python
  2. creates a private virtual environment in .venv, so nothing installed here
     touches their system Python or any other project
  3. installs dependencies once, then skips it on later runs by comparing a
     hash of requirements.txt
  4. asks for an Anthropic API key if .env does not have one
  5. picks a free port instead of failing when 8000 is busy
  6. starts the server and opens the browser once it is actually responding

Safe to run repeatedly. Everything after the first run is a few seconds.
=============================================================================
"""

import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REQS = os.path.join(HERE, "requirements.txt")
ENVF = os.path.join(HERE, ".env")
STAMP = os.path.join(VENV, ".deps_installed")

APP_NAME = "LaunchGuard AI"
PREFERRED_PORT = 8000
PORT_RANGE = 20
IS_WIN = platform.system() == "Windows"


# ---------------------------------------------------------------- output

def say(msg):
    print(f"  {msg}", flush=True)


def head(msg):
    print(f"\n{'=' * 62}\n  {msg}\n{'=' * 62}", flush=True)


def die(msg, detail=""):
    print(f"\n  PROBLEM: {msg}", flush=True)
    if detail:
        print(f"  {detail}", flush=True)
    print("\n  Press Enter to close.", flush=True)
    try:
        input()
    except EOFError:
        pass
    sys.exit(1)


# ---------------------------------------------------------------- python

def venv_python() -> str:
    return os.path.join(VENV, "Scripts" if IS_WIN else "bin",
                        "python.exe" if IS_WIN else "python")


def ensure_venv():
    """Create .venv if missing. Keeps this project isolated from everything else."""
    if os.path.isfile(venv_python()):
        return
    head("First run — setting up a private environment")
    say("This happens once and takes a minute or two.")
    try:
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    except subprocess.CalledProcessError:
        die("Could not create the virtual environment.",
            "On Debian/Ubuntu you may need: sudo apt install python3-venv")
    say("Environment created.")


def reqs_hash() -> str:
    if not os.path.isfile(REQS):
        return ""
    return hashlib.sha1(open(REQS, "rb").read()).hexdigest()


def ensure_deps():
    """Install requirements, but only when they have actually changed."""
    want = reqs_hash()
    if not want:
        die("requirements.txt is missing.",
            f"It should sit next to this file, in {HERE}")

    have = ""
    if os.path.isfile(STAMP):
        have = open(STAMP, encoding="utf-8").read().strip()
    if have == want:
        say("Dependencies already installed.")
        return

    head("Installing dependencies")
    say("First run downloads a few hundred MB. Later runs skip this.")
    py = venv_python()
    subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"],
                   check=False)
    r = subprocess.run([py, "-m", "pip", "install", "-r", REQS])
    if r.returncode != 0:
        die("Dependency installation failed.",
            "Check your internet connection and run this again.")
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(want)
    say("Dependencies installed.")


# ---------------------------------------------------------------- api key

def ensure_api_key():
    """
    The app needs an Anthropic key to write personas and run simulations.
    Everything else — upload, column mapping, clustering — works without one,
    so a missing key is a warning rather than a hard stop.
    """
    key = ""
    if os.path.isfile(ENVF):
        for line in open(ENVF, encoding="utf-8"):
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                key = line.split("=", 1)[-1].strip().strip('"').strip("'")
    if key:
        say(f"API key found ({key[:10]}…).")
        return

    head("Anthropic API key needed")
    print("""
  Personas and simulations call the Anthropic API, so a key is required for
  those. Uploading, column mapping and clustering all work without one.

  Get a key at  https://console.anthropic.com/settings/keys

  Paste it below, or press Enter to skip for now.
""", flush=True)
    try:
        entered = input("  Key: ").strip()
    except EOFError:
        entered = ""

    if not entered:
        say("Skipping. Add it to .env later and restart.")
        if not os.path.isfile(ENVF):
            with open(ENVF, "w", encoding="utf-8") as f:
                f.write("ANTHROPIC_API_KEY=\n")
        return

    lines = []
    if os.path.isfile(ENVF):
        lines = [l for l in open(ENVF, encoding="utf-8")
                 if not l.strip().startswith("ANTHROPIC_API_KEY")]
    lines.append(f"ANTHROPIC_API_KEY={entered}\n")
    with open(ENVF, "w", encoding="utf-8") as f:
        f.writelines(lines)
    say("Saved to .env")


# ---------------------------------------------------------------- port

def free_port(start=PREFERRED_PORT, span=PORT_RANGE) -> int:
    """First port nothing else is holding. Avoids the 'address in use' dead end."""
    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    die(f"No free port between {start} and {start + span}.")


def wait_until_up(port: int, timeout=90) -> bool:
    """Poll /api/health so the browser opens on a working page, not an error."""
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.6)
    return False


# ---------------------------------------------------------------- main

def main():
    head(f"{APP_NAME}")

    if sys.version_info < (3, 9):
        die(f"Python {sys.version_info.major}.{sys.version_info.minor} is too old.",
            "Install Python 3.10 or newer from python.org")

    if not os.path.isfile(os.path.join(HERE, "main.py")):
        die("main.py not found.",
            f"Put this launcher in the same folder as main.py. Currently in {HERE}")

    ensure_venv()
    ensure_deps()
    ensure_api_key()

    port = free_port()
    if port != PREFERRED_PORT:
        say(f"Port {PREFERRED_PORT} was busy — using {port} instead.")

    url = f"http://127.0.0.1:{port}"
    head("Starting the server")
    say(f"Address: {url}")
    say("Leave this window open. Close it to stop the app.")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [venv_python(), "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=HERE, env=env)

    try:
        if wait_until_up(port):
            say("Server is up. Opening your browser…")
            webbrowser.open(url)
        else:
            say("Server did not respond in time — check the messages above.")
        proc.wait()
    except KeyboardInterrupt:
        say("\nShutting down…")
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n  Stopped. Press Enter to close.", flush=True)
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        die(f"Unexpected error: {type(e).__name__}: {e}")
