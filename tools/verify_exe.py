"""Check that a Windows build opens and optionally runs actual image inference."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("dist/PlateWatch/PlateWatch.exe"))
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    directory = Path("test_artifacts")
    directory.mkdir(exist_ok=True)
    report = {"version": "2.2.0", "exe": str(args.exe.resolve()), "ok": False}
    try:
        run = subprocess.run([str(args.exe.resolve()), "--smoke-test"], timeout=30,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        report.update(gui_exit_code=run.returncode, stderr=run.stderr, ok=run.returncode == 0)
        if report["ok"] and args.image:
            ai_report = (directory / "frozen_ai_verification.json").resolve()
            run = subprocess.run([str(args.exe.resolve()), "--verify-image", str(args.image.resolve()),
                                  "--report", str(ai_report)], timeout=60, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace")
            report["ai_exit_code"] = run.returncode
            report["ok"] = run.returncode == 0
            if ai_report.exists():
                report["ai"] = json.loads(ai_report.read_text(encoding="utf-8"))
    except OSError as exc:
        report.update(ok=False, winerror=getattr(exc, "winerror", None), error=str(exc))
    except subprocess.TimeoutExpired:
        report.update(ok=False, error="Packaged application exceeded verification timeout")
    (directory / "exe_verification_2_2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
