"""Fine-tune a plate detector on a labelled YOLO dataset."""
import argparse
from pathlib import Path
import shutil
from platewatch.settings import APP_DIR, DATA_DIR
from platewatch.recognition import ensure_detector


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for Mongolian number plates")
    parser.add_argument("--data", type=Path, required=True, help="Roboflow/CVAT YOLO data.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or 0 for CUDA")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Dataset data.yaml not found")
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs and batch must be positive")
    from ultralytics import YOLO
    model = YOLO(args.model or str(ensure_detector({}, print)))
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0, fliplr=0.0, flipud=0.0,
                project=str(DATA_DIR / "training"), name="mongolian_plates", exist_ok=False)
    best = Path(model.trainer.best)
    destination = APP_DIR / "models" / "mongolian_best.pt"
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(best, destination)
    print("Fine-tuned model:", destination)
    print("Select this .pt file in PlateWatch Settings.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
