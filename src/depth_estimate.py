#!/usr/bin/env python3
"""
YOLO detections + distance/angle + GPS + BioCLIP 1 full Tree-of-Life classification.

Dependencies:
    pip install ultralytics pybioclip pillow

Usage example:
    python scan.py \
        --weights yolov8n.pt \
        --images ./photos \
        --hfov-deg 60 \
        --target-height-m 0.9 \
        --camera-lat 40.123 \
        --camera-lon -83.456 \
        --camera-bearing-deg 270 \
        --csv-out ./results/detections.csv \
        --rank species \
        --top-k 3
"""

import os
import glob
import math
import argparse
import csv
import datetime
import tempfile

# ---------------------------------------------------------------------------
# Site TW06 — Zebra Shelter, facing North
# GPS: 39°50.1224′N, 081°44.3053′W   SD: SD04
# Deploy: 2025-08-04 0950   Retrieve: 2025-08-11
# Camera: Trail Camera G600
# Notes: replaced SD card; 31.84 GB on disk; battery 100%
# ---------------------------------------------------------------------------
SITE_ID          = "TW06"
CAMERA_ID        = "SD04"
CAMERA_LAT       = 39.835373          # 39°50.1224′ N
CAMERA_LON       = -81.738422         # 081°44.3053′ W
CAMERA_BEARING   = 0.0                # North
HFOV_DEG         = 42.0               # Trail Camera G600 — verify in manual if needed
TARGET_HEIGHT_M  = 0.9                # ~typical deer shoulder height

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from bioclip import CustomLabelsClassifier

# ---------------------------------------------------------------------------
# Predefined species list for BioCLIP classification
# ---------------------------------------------------------------------------
SPECIES_LIST = [
    "Pere David's deer",
    "Bison",
    "Giraffe",
    "Rhinoceros",
    "Grevy's zebra",
    "Wild horse",
    "Sichuan takin",
    "Onager",
    "Scimitar-horned oryx",
    "Ostrich",
    "Fish",
]

# Allow loading of older YOLOv8 .pt files under PyTorch 2.5+
torch.serialization.add_safe_globals([DetectionModel])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO detections + distance/angle + GPS + BioCLIP 1 full Tree-of-Life labels"
    )

    # Paths / model
    p.add_argument("--weights",  default="yolov8n.pt", help="Path to YOLO weights (default: yolov8n.pt)")
    p.add_argument("--images",   required=True, help="Directory containing images (searched recursively)")
    p.add_argument("--device",   default="cpu",  help='Device: "cpu", "cuda:0", or "mps"')

    # YOLO inference params
    p.add_argument("--conf",   type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz",  type=int,   default=640,  help="YOLO inference image size")

    # Image cap / print control
    p.add_argument("--max-images",    type=int, default=0,  help="Cap on images to process (0 = no cap)")
    p.add_argument(
        "--max-det-print", type=int, default=-1,
        help="Max detections per image to print to console (0 = none, -1 = all)"
    )

    # BioCLIP 1 settings
    p.add_argument(
        "--top-k", type=int, default=1,
        help="Number of top BioCLIP predictions to record per detection (default: 1)"
    )

    # Geometry for distance/angle
    p.add_argument("--hfov-deg",        type=float, default=HFOV_DEG,
                   help=f"Camera HFOV in degrees (default: {HFOV_DEG})")
    p.add_argument("--target-height-m", type=float, default=TARGET_HEIGHT_M,
                   help=f"Approximate real-world height of target in metres (default: {TARGET_HEIGHT_M})")

    # Camera GPS & direction
    p.add_argument("--camera-lat",          type=float, default=CAMERA_LAT,
                   help=f"Camera latitude in decimal degrees (default: {CAMERA_LAT})")
    p.add_argument("--camera-lon",          type=float, default=CAMERA_LON,
                   help=f"Camera longitude in decimal degrees (default: {CAMERA_LON})")
    p.add_argument("--camera-bearing-deg",  type=str,   default=str(CAMERA_BEARING),
                   help=f'Camera bearing degrees clockwise from North (default: {CAMERA_BEARING})')

    # Output
    p.add_argument("--csv-out", type=str,
                   default=f"MakeOHI-O-2026/results_{SITE_ID}_{CAMERA_ID}.csv",
                   help=f"Path to output CSV file (default: MakeOHI-O-2026/results_{SITE_ID}_{CAMERA_ID}.csv)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_camera_bearing(val: str):
    """Return float in [0, 360) or None if value is NA / blank."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s) % 360.0
    except Exception:
        return None


def parse_iso_timestamp_from_filename(path_or_name: str):
    """
    Parse timestamp embedded in filenames like:
        NSCF0001_231031200001_0088.JPG
                 YYMMDDHHMMSS
    Returns ISO string or None.
    """
    try:
        base  = os.path.basename(path_or_name)
        parts = base.split("_")
        if len(parts) < 2:
            return None
        dt = parts[1]
        if len(dt) < 12 or not dt[:12].isdigit():
            return None
        return datetime.datetime(
            int("20" + dt[0:2]), int(dt[2:4]),  int(dt[4:6]),
            int(dt[6:8]),        int(dt[8:10]), int(dt[10:12])
        ).isoformat()
    except Exception:
        return None


def estimate_distance_and_angle(img_w, img_h, x_min, y_min, x_max, y_max, hfov_deg, real_h_m):
    """Return (distance_m, horizontal_angle_deg) from a bounding box."""
    bbox_h = y_max - y_min
    if bbox_h <= 0:
        return float("inf"), 0.0
    hfov_rad = math.radians(hfov_deg)
    fx       = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
    dist_m   = (real_h_m * fx) / bbox_h
    dx       = ((x_min + x_max) / 2.0) - (img_w / 2.0)
    angle_deg = math.degrees(math.atan(dx / fx))
    return dist_m, angle_deg


def project_gps(lat, lon, distance_m, bearing_deg):
    """Vincenty-style forward projection on a spherical earth."""
    R        = 6378137.0
    lat1     = math.radians(lat)
    lon1     = math.radians(lon)
    bearing  = math.radians(bearing_deg)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def gather_images_recursive(root_dir: str):
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", e), recursive=True))
    return sorted(set(paths))


def format_prediction(pred: dict) -> tuple[str, float]:
    """Extract label and score from a CustomLabelsClassifier prediction dict."""
    label = pred.get("classification", "unknown")
    score = float(pred.get("score", 0.0))
    return label, score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    assert os.path.isfile(args.weights) or os.path.isdir(args.weights), f"Weights not found: {args.weights}"
    assert os.path.isdir(args.images),   f"Images directory not found: {args.images}"

    camera_bearing = parse_camera_bearing(args.camera_bearing_deg)

    paths = gather_images_recursive(args.images)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")
    if args.max_images > 0:
        paths = paths[: args.max_images]

    # ── Load YOLO ──────────────────────────────────────────────────────────
    print(f"Loading YOLO model: {args.weights}")
    yolo_model = YOLO(args.weights)

    # ── Load BioCLIP 1 via pybioclip ───────────────────────────────────────
    print(f"Loading BioCLIP 1 CustomLabelsClassifier ({len(SPECIES_LIST)} species, top-k={args.top_k})...")
    classifier = CustomLabelsClassifier(
        SPECIES_LIST,
        device=args.device,
        model_str="hf-hub:imageomics/bioclip",
    )
    print("BioCLIP 1 ready.")

    # ── Output dirs ────────────────────────────────────────────────────────
    csv_dir = os.path.dirname(args.csv_out)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    csv_rows = []

    print(f"\nStarting scan on {len(paths)} image(s)")
    print(f"Distance model: target height={args.target_height_m} m, HFOV={args.hfov_deg} deg")
    bearing_str = f"{camera_bearing} deg" if camera_bearing is not None else "NA"
    print(f"Camera: lat={args.camera_lat}, lon={args.camera_lon}, bearing={bearing_str}\n")

    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        print(f"[{i}/{len(paths)}] {fname}")

        # Open image
        try:
            img = Image.open(p).convert("RGB")
            img_w, img_h = img.size
            try:
                file_timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat()
            except Exception:
                file_timestamp = None
        except Exception as e:
            print(f"  [SKIP] Could not open: {e}")
            continue

        timestamp_from_name = parse_iso_timestamp_from_filename(fname)

        # ── YOLO inference ─────────────────────────────────────────────────
        results = yolo_model.predict(
            source=p,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        kept_det_idx  = 0

        for det in results[0].boxes:
            x_min, y_min, x_max, y_max = [float(v) for v in det.xyxy[0]]
            yolo_conf = float(det.conf[0])

            # Clamp crop
            x0 = max(0, int(x_min));  y0 = max(0, int(y_min))
            x1 = min(img_w, int(x_max)); y1 = min(img_h, int(y_max))
            if x1 <= x0 or y1 <= y0:
                continue

            crop = img.crop((x0, y0, x1, y1))

            # ── BioCLIP 1 classification ───────────────────────────────────
            # pybioclip needs a file path, so save the crop to a temp file.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            try:
                crop.save(tmp_path, format="JPEG")
                predictions = classifier.predict(tmp_path, k=args.top_k)
            finally:
                os.unlink(tmp_path)

            # predictions is a list of dicts, best first
            top_pred  = predictions[0] if predictions else {}
            top_label, top_score = format_prediction(top_pred)

            kept_det_idx += 1

            # ── Geometry ───────────────────────────────────────────────────
            d_m, a_deg = estimate_distance_and_angle(
                img_w, img_h, x_min, y_min, x_max, y_max,
                args.hfov_deg, args.target_height_m
            )

            if camera_bearing is None:
                absolute_bearing          = None
                animal_lat, animal_lon    = args.camera_lat, args.camera_lon
            else:
                absolute_bearing          = (camera_bearing + a_deg) % 360.0
                animal_lat, animal_lon    = project_gps(
                    args.camera_lat, args.camera_lon, d_m, absolute_bearing
                )

            # ── Console output ─────────────────────────────────────────────
            if args.max_det_print != 0 and (args.max_det_print < 0 or kept_det_idx <= args.max_det_print):
                abs_str = f"{absolute_bearing:.1f} deg" if absolute_bearing is not None else "NA"
                print(
                    f"  #{kept_det_idx} {top_label} | "
                    f"yolo={yolo_conf:.2f} | bio={top_score:.3f} | "
                    f"d={d_m:.1f} m | rel={a_deg:+.1f} deg | abs={abs_str}"
                )

            # ── Build CSV row ──────────────────────────────────────────────
            row = {
                "image":                       fname,
                "image_path":                  p,
                "timestamp_file":              file_timestamp,
                "timestamp_from_name":         timestamp_from_name,
                "detection_index":             kept_det_idx,
                # Top BioCLIP prediction
                "bioclip_label":               top_label,
                "bioclip_score":               top_score,
                # YOLO
                "yolo_conf":                   yolo_conf,
                "xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max,
                # Geometry / GPS
                "distance_m":                  d_m,
                "angle_deg":                   a_deg,
                "camera_lat":                  args.camera_lat,
                "camera_lon":                  args.camera_lon,
                "camera_bearing_deg":          camera_bearing if camera_bearing is not None else "NA",
                "animal_lat":                  animal_lat,
                "animal_lon":                  animal_lon,
                "animal_absolute_bearing_deg": absolute_bearing if absolute_bearing is not None else "NA",
            }

            # Optionally add runner-up predictions as extra columns
            for rank_idx, pred in enumerate(predictions[1:], start=2):
                lbl, sc = format_prediction(pred)
                row[f"bioclip_label_{rank_idx}"] = lbl
                row[f"bioclip_score_{rank_idx}"] = sc

            csv_rows.append(row)


    # ── Write CSV ──────────────────────────────────────────────────────────
    if csv_rows:
        # Build fieldnames from the union of all row keys (preserving order)
        fieldnames = list(dict.fromkeys(k for row in csv_rows for k in row))

        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nDetections saved to: {args.csv_out}")
    else:
        print("\nNo detections found — CSV not written.")


if __name__ == "__main__":
    main()
