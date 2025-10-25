import os
from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


def _resolve_cascade_path(cascade_file: Optional[str] = None) -> str:
    """Return an absolute path to the cascade xml bundled with this package.

    If cascade_file is given and exists, use it. Otherwise, use the bundled
    'lbpcascade_animeface.xml' next to this file.
    """
    if cascade_file and os.path.isfile(cascade_file):
        return cascade_file
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(here, "lbpcascade_animeface.xml")
    if not os.path.isfile(default_path):
        raise RuntimeError(f"Anime face cascade not found: {default_path}")
    return default_path


def _detect_from_gray(gray: np.ndarray, cascade_file: Optional[str] = None,
                      scale_factor: float = 1.1, min_neighbors: int = 5,
                      min_size: Tuple[int, int] = (24, 24)) -> List[List[int]]:
    """Run detection on a grayscale uint8 image and return list of [x,y,w,h]."""
    cascade_path = _resolve_cascade_path(cascade_file)
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]


def detect_from_bgr_image(image_bgr: np.ndarray, cascade_file: Optional[str] = None,
                          scale_factor: float = 1.1, min_neighbors: int = 5,
                          min_size: Tuple[int, int] = (24, 24)) -> List[List[int]]:
    """Detect faces in a BGR image array and return a list of [x, y, w, h].

    Args:
        image_bgr: OpenCV-style BGR image (H, W, 3), uint8.
        cascade_file: Optional custom cascade path.
        scale_factor, min_neighbors, min_size: Cascade parameters.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return _detect_from_gray(gray, cascade_file, scale_factor, min_neighbors, min_size)


def detect(filename: str, cascade_file: Optional[str] = None,
           scale_factor: float = 1.1, min_neighbors: int = 5,
           min_size: Tuple[int, int] = (24, 24)) -> List[List[int]]:
    """Detect faces in an image file and return a list of [x, y, w, h].

    Returns an empty list when no faces are detected.
    """
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Image not found or cannot be read: {filename}")
    return detect_from_bgr_image(image, cascade_file, scale_factor, min_neighbors, min_size)


def pick_best_face(bboxes: Sequence[Sequence[int]], strategy: str = "largest") -> Optional[List[int]]:
    """Pick one face bbox from list.

    strategy="largest": choose the largest area bbox.
    strategy="leftmost": choose the leftmost bbox.
    strategy="center": choose the bbox closest to image center (if you pass center separately, preferred in crop).
    """
    if not bboxes:
        return None
    if strategy == "largest":
        return list(max(bboxes, key=lambda b: b[2] * b[3]))
    if strategy == "leftmost":
        return list(min(bboxes, key=lambda b: b[0]))
    # default fallback
    return list(bboxes[0])


def expand_to_square(x: int, y: int, w: int, h: int, margin: float = 0.2,
                     image_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """Expand a bbox to a square with margin. Margin is a ratio of the longer side.

    If image_size=(W,H) is given, clamp the square inside image bounds.
    Returns (x, y, side, side).
    """
    side = int(round(max(w, h) * (1.0 + margin)))
    cx = x + w // 2
    cy = y + h // 2
    sx = cx - side // 2
    sy = cy - side // 2
    if image_size is not None:
        W, H = image_size
        sx = max(0, min(sx, W - side))
        sy = max(0, min(sy, H - side))
        # If side exceeds bounds, shrink to fit
        side = min(side, W, H)
        sx = max(0, min(sx, W - side))
        sy = max(0, min(sy, H - side))
    return int(sx), int(sy), int(side), int(side)


def detect_and_crop(image_bgr: np.ndarray, cascade_file: Optional[str] = None,
                    pick: str = "largest", margin: float = 0.2,
                    output_size: Optional[int] = None) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Detect a face and return (cropped_bgr, bbox).

    - pick: strategy to pick bbox if multiple ("largest" or "leftmost").
    - margin: ratio added around face before squaring.
    - output_size: if given, resize crop to (output_size, output_size).
    Returns (crop, bbox) where bbox is (x, y, w, h) in the original image. If no face, returns (original, None).
    """
    bboxes = detect_from_bgr_image(image_bgr, cascade_file)
    best = pick_best_face(bboxes, pick)
    if best is None:
        return image_bgr, None
    x, y, w, h = best
    W, H = image_bgr.shape[1], image_bgr.shape[0]
    sx, sy, sw, sh = expand_to_square(x, y, w, h, margin=margin, image_size=(W, H))
    crop = image_bgr[sy:sy + sh, sx:sx + sw, :]
    if output_size is not None and output_size > 0:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop, (sx, sy, sw, sh)


class TemporalBoxSmoother:
    """Simple temporal smoothing for sequential bboxes.

    Use when processing frames to avoid jitter. Keeps an exponential moving average of bbox parameters.
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self.state: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h

    def update(self, bbox: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None:
            if self.state is None:
                return None
            sx, sy, sw, sh = self.state
            return (int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh)))
        x, y, w, h = map(float, bbox[:4])
        if self.state is None:
            self.state = (x, y, w, h)
        else:
            sx, sy, sw, sh = self.state
            ax = self.alpha * x + (1.0 - self.alpha) * sx
            ay = self.alpha * y + (1.0 - self.alpha) * sy
            aw = self.alpha * w + (1.0 - self.alpha) * sw
            ah = self.alpha * h + (1.0 - self.alpha) * sh
            self.state = (ax, ay, aw, ah)
        sx, sy, sw, sh = self.state
        return (int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh)))


# -------------------------
# Coordinates-only helpers
# -------------------------

def detect_largest_bbox_from_bgr(image_bgr: np.ndarray,
                                 cascade_file: Optional[str] = None,
                                 scale_factor: float = 1.1,
                                 min_neighbors: int = 5,
                                 min_size: Tuple[int, int] = (24, 24)) -> Optional[Tuple[int, int, int, int]]:
    """Detect and return the largest face bbox [x,y,w,h] or None.

    This enforces the "always use the largest face" policy.
    """
    boxes = detect_from_bgr_image(
        image_bgr,
        cascade_file=cascade_file,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    best = pick_best_face(boxes, strategy="largest")
    return None if best is None else (best[0], best[1], best[2], best[3])


def bbox_to_anchor(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int]
) -> dict:
    """Convert bbox to an anchor dict useful for downstream systems (e.g., Live2D).

    Returns a dict with pixel and normalized values:
      - bbox: (x, y, w, h) in pixels
      - center: (cx, cy) in pixels
      - center_norm: (cx/W, cy/H)
      - size: max(w, h) in pixels
      - size_norm: size / max(W, H)
      - image_size: (W, H)
    """
    x, y, w, h = bbox
    W, H = image_size
    cx = x + w / 2.0
    cy = y + h / 2.0
    size = float(max(w, h))
    max_side = float(max(W, H)) if max(W, H) > 0 else 1.0
    return {
        "bbox": (int(x), int(y), int(w), int(h)),
        "center": (float(cx), float(cy)),
        "center_norm": (float(cx) / float(W), float(cy) / float(H)) if W > 0 and H > 0 else (0.5, 0.5),
        "size": size,
        "size_norm": size / max_side,
        "image_size": (int(W), int(H)),
    }


def compute_face_coordinates(
    image_bgr: np.ndarray,
    cascade_file: Optional[str] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (24, 24),
) -> Optional[dict]:
    """Compute coordinates for the largest face without cropping.

    Returns a dict from bbox_to_anchor(...) or None if not detected.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    bbox = detect_largest_bbox_from_bgr(
        image_bgr,
        cascade_file=cascade_file,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    if bbox is None:
        return None
    H, W = image_bgr.shape[0], image_bgr.shape[1]
    return bbox_to_anchor(bbox, (W, H))


class LargestFaceTracker:
    """Track the largest face over time and return stable coordinates only.

    - Always picks the largest face in the current frame.
    - Applies temporal smoothing (EMA) to reduce jitter.
    - If detection fails in a frame, returns the last known (smoothed) coordinates.
    """

    def __init__(self, alpha: float = 0.7,
                 cascade_file: Optional[str] = None,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (24, 24)):
        self.smoother = TemporalBoxSmoother(alpha=alpha)
        self.cascade_file = cascade_file
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def update(self, image_bgr: np.ndarray) -> Optional[dict]:
        if image_bgr is None or image_bgr.size == 0:
            return None
        bbox = detect_largest_bbox_from_bgr(
            image_bgr,
            cascade_file=self.cascade_file,
            scale_factor=self.scale_factor,
            min_neighbors=self.min_neighbors,
            min_size=self.min_size,
        )
        smoothed = self.smoother.update(bbox)
        if smoothed is None:
            return None
        H, W = image_bgr.shape[0], image_bgr.shape[1]
        return bbox_to_anchor(smoothed, (W, H))


def _cli():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Anime face detection and optional crop")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--cascade", type=str, default=None, help="Path to cascade xml (optional)")
    parser.add_argument("--pick", type=str, default="largest", choices=["largest", "leftmost"],
                        help="Strategy to pick one face when multiple detected")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin ratio around face before squaring")
    parser.add_argument("--output-size", type=int, default=0, help="If > 0, resize crop to NxN and save if --out is set")
    parser.add_argument("--out", type=str, default=None, help="If set, save the cropped image here")
    parser.add_argument("--print-json", action="store_true", help="Print detected boxes as JSON to stdout")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Image not found: {args.image}")

    boxes = detect_from_bgr_image(image, args.cascade)
    best = pick_best_face(boxes, args.pick)

    if args.print_json:
        print(json.dumps({"boxes": boxes, "best": best}))

    if args.out is not None:
        out_size = args.output_size if args.output_size and args.output_size > 0 else None
        crop, _ = detect_and_crop(image, args.cascade, pick=args.pick, margin=args.margin, output_size=out_size or 0)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, crop)


if __name__ == "__main__":
    _cli()

