"""Find and rectify light plate quadrilaterals when the learned detector misses.

These are geometric candidates, not YOLO predictions. Recognition must confirm
their complete text; no characters are added from dimensions or plate series.
"""


def plate_quadrilaterals(frame, limit=3, minimum_width=40):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    scale = min(1, 1280 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour) / scale ** 2
        if not max(500, height * width * .001) <= area <= height * width * .9:
            continue
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32) / scale
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        quad = np.array([points[sums.argmin()], points[differences.argmin()],
                         points[sums.argmax()], points[differences.argmax()]], np.float32)
        if len(np.unique(quad, axis=0)) != 4:
            continue
        w = round(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = round(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        if w < minimum_width or h < 18 or not 1.1 <= w / h <= 7:
            continue
        candidates.append((area, quad, w, h))
    candidates.sort(key=lambda item: item[0], reverse=True)
    results = []
    for _, quad, w, h in candidates[:max(12, limit * 4)]:
        crop_scale = min(1, 1280 / max(w, h))
        w, h = max(2, round(w * crop_scale)), max(2, round(h * crop_scale))
        destination = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        crop = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, destination), (w, h))
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        components, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        characters = [cv2.boundingRect(c) for c in components]
        characters = [(x, y, a, b) for x, y, a, b in characters
                      if .15 * h <= b <= .9 * h and .01 * w <= a <= .25 * w]
        if len(characters) < 5 or not .035 <= float((ink > 0).mean()) <= .65:
            continue
        bbox = (max(0, int(quad[:, 0].min())), max(0, int(quad[:, 1].min())),
                min(width, int(np.ceil(quad[:, 0].max())) + 1), min(height, int(np.ceil(quad[:, 1].max())) + 1))
        results.append((bbox, crop))
        if len(results) >= limit:
            break
    return results
