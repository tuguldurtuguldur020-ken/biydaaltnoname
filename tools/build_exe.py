"""Build a folder containing PlateWatch.exe and its Python dependencies."""
import shutil
import subprocess
import sys
from platewatch import __version__
from platewatch.settings import APP_DIR, DATA_DIR


def main():
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
            "--name", "PlateWatch", "--collect-data", "easyocr", "--collect-data", "ultralytics",
            "--hidden-import", "easyocr.model.vgg_model",
            "--hidden-import", "PIL._tkinter_finder", "--hidden-import", "torchvision",
            "--exclude-module", "polars", "--exclude-module", "_polars_runtime_32",
            "--exclude-module", "easyocr.DBNet", "--exclude-module", "easyocr.detection_db",
            "--exclude-module", "easyocr.detection", "--exclude-module", "easyocr.craft",
            "--specpath", str(APP_DIR / "build" / "specs"),
            "--add-data", str(APP_DIR / "README.md") + ";.",
            "--add-data", str(APP_DIR / "docs") + ";docs", str(APP_DIR / "biydaalt.py")]
    subprocess.run(args, cwd=APP_DIR, check=True)
    destination = APP_DIR / "dist" / "PlateWatch"
    if (APP_DIR / "models").exists():
        shutil.copytree(APP_DIR / "models", destination / "models", dirs_exist_ok=True)
    cache = DATA_DIR / "models"
    if cache.exists():
        shutil.copytree(cache, destination / "data" / "models", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("craft_mlt_25k.pth", "user_network", "*.download"))
    shutil.copy2(APP_DIR / "README.md", destination / "README.md")
    shutil.copytree(APP_DIR / "docs", destination / "docs", dirs_exist_ok=True)
    (destination / "VERSION.txt").write_text(f"PlateWatch {__version__}\nPython desktop GUI / YOLO + Mongolian OCR\n", encoding="utf-8")
    print("Built:", destination / "PlateWatch.exe")
    print("Share the entire dist/PlateWatch folder, including _internal and models.")


if __name__ == "__main__":
    main()
