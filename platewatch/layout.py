"""Locate and straighten actual character rows inside a detected plate crop."""


def _row_pairs(binary):
    import cv2
    height, width = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    for contour in contours:
        x, y, a, b = cv2.boundingRect(contour)
        if (x > 1 and y > 1 and x + a < width - 1 and y + b < height - 1
                and .08 * height <= b <= .60 * height and .16 <= a / b <= 1.1
                and a <= .30 * width and cv2.contourArea(contour) >= .0007 * width * height):
            characters.append((x, y, a, b, contour))
    rows = []
    for char in sorted(characters, key=lambda c: c[1] + c[3] / 2):
        center = char[1] + char[3] / 2
        for row in rows:
            middle = sum(c[1] + c[3] / 2 for c in row) / len(row)
            row_height = max(c[3] for c in row)
            if .5 <= char[3] / row_height <= 2 and abs(center - middle) <= .5 * max(char[3], row_height):
                row.append(char)
                break
        else:
            rows.append([char])
    groups = []
    for row in rows:
        tallest = max(c[3] for c in row)
        ordered = sorted((c for c in row if c[3] >= tallest * .55), key=lambda c: c[0])
        current = []
        for char in ordered:
            if current and char[0] - max(c[0] + c[2] for c in current) > .9 * tallest:
                groups.append(current)
                current = []
            current.append(char)
        if current:
            groups.append(current)
    pairs = []
    for top in (g for g in groups if len(g) == 4):
        tx1, tx2 = min(c[0] for c in top), max(c[0] + c[2] for c in top)
        ty = sum(c[1] + c[3] / 2 for c in top) / 4
        th = max(c[3] for c in top)
        for bottom in (g for g in groups if len(g) == 3):
            bx1, bx2 = min(c[0] for c in bottom), max(c[0] + c[2] for c in bottom)
            by = sum(c[1] + c[3] / 2 for c in bottom) / 3
            overlap = max(0, min(tx2, bx2) - max(tx1, bx1)) / max(1, min(tx2 - tx1, bx2 - bx1))
            if .65 * th <= by - ty <= 2.5 * th and overlap >= .65:
                pairs.append((top, bottom))
    return pairs


def two_row_crops(crop):
    """Yield bounded number/letter row images; never fill missing characters.

    Local contrast removes the car body's lighting gradient. Character-shaped
    components, their row alignment and the 4+3 layout keep bumper trim out of
    OCR, even when a detector includes a large border around a square plate.
    """
    import cv2
    import numpy as np
    h, w = crop.shape[:2]
    if h < 18 or w < 24 or not .7 <= w / h <= 2.8:
        return
    scale = min(3, 360 / h, 720 / w)
    image = cv2.resize(crop, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height = gray.shape[0]
    kernel = tuple(max(low, round(height * ratio) | 1) for low, ratio in ((9, .06), (13, .09)))
    ink = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    pairs = _row_pairs(cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    if not pairs:
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        pairs = _row_pairs(binary)
        if not pairs:
            # A one-pixel bridge after resizing can join adjacent characters.
            # Shrink only the segmentation mask; OCR still sees original pixels.
            pairs = _row_pairs(cv2.erode(binary, np.ones((2, 2), np.uint8)))
    for top, bottom in pairs[:2]:
        result = []
        for row in (top, bottom):
            points = np.concatenate([c[4] for c in row])
            center, size, angle = cv2.minAreaRect(points)
            pad = max(3, max(c[3] for c in row) * .12)
            box = cv2.boxPoints((center, (size[0] + 2 * pad, size[1] + 2 * pad), angle))
            sums, diff = box.sum(axis=1), np.diff(box, axis=1).ravel()
            quad = np.array([box[sums.argmin()], box[diff.argmin()], box[sums.argmax()], box[diff.argmax()]], np.float32)
            if len(np.unique(quad, axis=0)) != 4:
                break
            a = max(8, round(np.linalg.norm(quad[1] - quad[0])))
            b = max(8, round(np.linalg.norm(quad[3] - quad[0])))
            target = np.float32([(0, 0), (a - 1, 0), (a - 1, b - 1), (0, b - 1)])
            result.append(cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, target), (a, b), borderMode=cv2.BORDER_REPLICATE))
        if len(result) == 2:
            yield tuple(result)
