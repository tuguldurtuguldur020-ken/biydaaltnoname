"""Download/check the plate detector and Mongolian OCR weights."""
from platewatch.recognition import PlateRecognizer
from platewatch.settings import load_settings


def main():
    recognizer = PlateRecognizer(load_settings(), print)
    print("Ready. YOLO classes:", recognizer.model.names)
    return recognizer


if __name__ == "__main__":
    main()
