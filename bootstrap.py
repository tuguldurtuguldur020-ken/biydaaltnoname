"""Prepare a project-local Python environment and launch without an editor."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
ENV_PYTHON = ENV / "Scripts" / "python.exe"
MODULES = ["cv2", "numpy", "PIL", "ultralytics", "easyocr", "torch", "torchvision", "onnxruntime", "yaml"]


def main():
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        print("Install Python 3.11, 3.12 or 3.13 (64-bit), then run run.bat again.")
        return 1
    temp = ROOT / ".setup_tmp"
    temp.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(TEMP=str(temp), TMP=str(temp), PYTHONUTF8="1")
    if not ENV_PYTHON.exists():
        print("Creating .venv...", flush=True)
        result = subprocess.run([sys.executable, "-m", "venv", str(ENV)], env=environment)
        if result.returncode:
            return result.returncode
    check = "import importlib.util; raise SystemExit(not all(importlib.util.find_spec(n) for n in " + repr(MODULES) + "))"
    ready = subprocess.run([str(ENV_PYTHON), "-c", check], env=environment).returncode == 0
    if not ready or "--install" in sys.argv:
        # Repair an incomplete first installation, without touching global packages.
        if subprocess.run([str(ENV_PYTHON), "-m", "pip", "--version"], env=environment, stdout=subprocess.DEVNULL).returncode:
            subprocess.run([str(ENV_PYTHON), "-m", "ensurepip", "--upgrade"], env=environment, check=True)
        print("Installing dependencies (first run may take several minutes)...", flush=True)
        result = subprocess.run([str(ENV_PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], env=environment)
        if result.returncode:
            return result.returncode
    if "--install" in sys.argv:
        return 0
    args = [arg for arg in sys.argv[1:] if arg != "--launch"]
    print("Opening PlateWatch...", flush=True)
    return subprocess.run([str(ENV_PYTHON), "-X", "utf8", str(ROOT / "biydaalt.py"), *args], cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
