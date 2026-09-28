"""
Detect and remove the ruler/label region from true-negative images using
OWLv2 zero-shot object detection, so the classifier stops keying on it.

Unlike the saturation heuristic (smart_crop_ruler.py), this doesn't assume
anything about the artifact's color -- it directly detects "ruler" /
"scale bar" / "label" via text prompts and paints those regions out.
Useful as a fallback for images where the color heuristic fails (dark or
low-saturation corrosion), or as the primary method if your dataset has
varied backgrounds.

Install:
    pip install transformers torch pillow opencv-python accelerate

Usage:
    python owlv2_ruler_removal.py --input ./tn_images --output ./tn_masked --review ./tn_review
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

QUERIES = [
    "a ruler",
    "a measuring scale with cm markings",
    "a metric ruler",
    "a paper label with a catalog number",
    "measuring tool", "calibration chart","scale bar"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    ).to(DEVICE)
    model.eval()
    return processor, model


def detect_ruler_boxes(image_pil, processor, model, score_thresh=0.15):
    inputs = processor(text=[QUERIES], images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]]).to(DEVICE)  # (h, w)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, threshold=score_thresh, target_sizes=target_sizes, text_labels=[QUERIES]
    )[0]

    boxes = results["boxes"].cpu().numpy()  # x0, y0, x1, y1
    scores = results["scores"].cpu().numpy()
    labels = results["text_labels"]
    return list(zip(boxes, scores, labels))


def mask_boxes(img_bgr, detections, pad_frac=0.02, fill_color=(255, 255, 255)):
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for box, score, label in detections:
        x0, y0, x1, y1 = box
        pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        x0 = max(0, int(x0 - pad_x))
        y0 = max(0, int(y0 - pad_y))
        x1 = min(w, int(x1 + pad_x))
        y1 = min(h, int(y1 + pad_y))
        out[y0:y1, x0:x1] = fill_color
    return out


def process_folder(input_dir: Path, output_dir: Path, review_dir: Path, score_thresh=0.15):
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model()

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    masked_count, flagged = 0, 0
    for p in paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"[skip] could not read {p.name}")
            continue
        image_pil = Image.open(p).convert("RGB")

        detections = detect_ruler_boxes(image_pil, processor, model, score_thresh)

        if not detections:
            # Nothing found -- could mean no ruler present, or a miss.
            # Flag for manual check rather than silently leaving it untouched.
            cv2.imwrite(str(review_dir / p.name), img_bgr)
            flagged += 1
            continue

        masked = mask_boxes(img_bgr, detections)
        cv2.imwrite(str(output_dir / p.name), masked)
        masked_count += 1

        # Annotated copy in review folder so you can audit detection quality.
        annotated = img_bgr.copy()
        for box, score, label in detections:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(
                annotated, f"{label}:{score:.2f}", (x0, max(0, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
        cv2.imwrite(str(review_dir / p.name), annotated)

    print(f"Processed {len(paths)} images: {masked_count} masked, {flagged} flagged (no detection).")
    print(f"Review copies saved to {review_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--score-thresh", type=float, default=0.15)
    args = parser.parse_args()
    process_folder(args.input, args.output, args.review, args.score_thresh)
