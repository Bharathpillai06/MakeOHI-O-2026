# MakeOHI-O-2026
# WildSense

**AI-Powered Wildlife Camera Trap System**

PIR Motion Detection • YOLOv8 Animal Detection • BioCLIP Species Identification • GPS Animal Location Estimation

WildSense is an embedded wildlife monitoring system that automatically detects animals, identifies species, and estimates their GPS location using computer vision and geometric projection. The system transforms traditional camera traps into intelligent ecological sensors capable of generating structured datasets for conservation research.

Developed for **Make I/O 2026**.

---

## Table of Contents

- [Overview](#overview)
- [Inspiration](#inspiration)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Decision Logic](#decision-logic)
- [Outputs](#outputs)
- [CSV Output Schema](#csv-output-schema)
- [Full Pipeline Example](#full-pipeline-example)
- [GPS Projection Model](#gps-projection-model)
- [BioCLIP Species Classification](#bioclip-species-classification)
- [Serial Communication Protocol](#serial-communication-protocol)
- [Repository Structure](#repository-structure)
- [Success Metrics](#success-metrics)
- [Non-Goals](#non-goals)
- [Long-Term Vision](#long-term-vision)

---

# Overview

Traditional camera traps capture thousands of images but provide **no automatic species identification or location data**, requiring researchers to manually review images.

**WildSense automates this process** by combining motion-triggered capture, AI-based detection and classification, and GPS location estimation directly at the field device.

### Key Characteristics

- Event-driven (PIR triggered)
- No continuous inference
- No required cloud connection
- Local SD card storage on ESP32-CAM
- Structured CSV output per scan batch
- Geometric GPS projection from bounding box geometry

> [!IMPORTANT]
> All inference runs locally on the Raspberry Pi. No cloud dependency is required.

---

# Inspiration

WildSense was inspired by the **SmartWilds dataset**, a multimodal wildlife monitoring project conducted in **Summer 2025 at The Wilds safari park in Ohio**.

SmartWilds collected synchronized:

- Drone imagery
- Camera trap images and videos
- Bioacoustic recordings

During this project we observed how much **manual processing is required to analyze camera trap data**, which motivated the creation of WildSense to automate species identification and spatial analysis.

---

# System Architecture

## Hardware

- ESP32-CAM module (AI Thinker)
- PIR motion sensor (GPIO 13)
- MicroSD card (onboard ESP32-CAM)
- Raspberry Pi (any model with UART)
- UART serial connection (115200 baud)

---

## Two-Device Architecture

```
┌─────────────────────────┐         UART Serial         ┌────────────────────────────┐
│      ESP32-CAM          │  ──────────────────────────► │      Raspberry Pi          │
│                         │                              │                            │
│  • PIR motion trigger   │   IMG:<filename>:<size>\n    │  • Receives images         │
│  • JPEG capture         │   [raw JPEG bytes]           │  • Batches every 10 hours  │
│  • MicroSD storage      │   END\n                      │  • Runs YOLOv8 detection   │
│  • UART transmission    │                              │  • Runs BioCLIP classify   │
│  • 1-min cooldown       │ ◄──────────────────────────  │  • GPS projection          │
│  • CLEAR command resp.  │        CLEAR\n               │  • CSV output              │
└─────────────────────────┘                              └────────────────────────────┘
```

### Camera Node (ESP32-CAM)

Captures wildlife images in the field.

Features:
- ESP32-CAM module (AI Thinker pinout)
- PIR motion sensor
- 1600×1200 JPEG capture (UXGA with PSRAM)
- MicroSD storage via SD_MMC
- 1-minute capture cooldown
- UART transmission to Raspberry Pi

### Processing Node (Raspberry Pi)

Runs AI inference on accumulated images.

Features:
- Receives images over UART serial
- Accumulates images for 10-hour batch windows
- YOLOv8 object detection
- BioCLIP species classification
- Distance + bearing geometry
- GPS coordinate projection
- CSV results output

---

## Software Stack

- Python 3.9+
- PyTorch
- Ultralytics YOLOv8
- pybioclip (`hf-hub:imageomics/bioclip`)
- Pillow
- Arduino / ESP-IDF (ESP32-CAM firmware)

---

# Installation

```bash
pip install ultralytics
pip install pybioclip pillow
pip install torch torchvision
```

---

# Pipeline Workflow

## Step 1: Motion Trigger (ESP32-CAM)

When PIR sensor detects motion:

1. Check 1-minute cooldown timer
2. Capture JPEG image
3. Save to MicroSD card
4. Transmit image over UART to Raspberry Pi
5. Return to idle

> [!TIP]
> A 1-minute cooldown prevents burst captures from a single animal event.

---

## Step 2: Image Accumulation (Raspberry Pi)

The Raspberry Pi receiver thread continuously listens on `/dev/ttyS0` and:

- Parses `IMG:<filename>:<size>` headers
- Reads raw JPEG bytes
- Validates `END` footer
- Saves with timestamp prefix to `./received_images/`

---

## Step 3: Batch Scan (Every 10 Hours)

```bash
python scan.py \
  --weights yolov8s_openvino_model \
  --images ./received_images \
  --csv-out ./results/detections_<timestamp>.csv \
  --device cpu
```

The scanner:

1. Loads all accumulated `.jpg` files
2. Runs YOLOv8 detection on each image
3. Crops each detected bounding box
4. Passes crop to BioCLIP for species classification
5. Computes distance and bearing from bounding box geometry
6. Projects GPS coordinates of the animal
7. Writes results to CSV

---

## Step 4: Object Detection (YOLOv8)

For each image:

- Bounding boxes extracted from YOLO results
- Confidence score retained
- Crop region passed to BioCLIP

---

## Step 5: Species Classification (BioCLIP)

Each detected crop is classified against a predefined species list:

```python
SPECIES_LIST = [
    "Pere David's deer", "Bison", "Giraffe", "Rhinoceros",
    "Grevy's zebra", "Wild horse", "Sichuan takin", "Onager",
    "Scimitar-horned oryx", "Ostrich", "Fish",
]
```

Top-K predictions are recorded per detection.

---

## Step 6: GPS Projection

Distance and bearing are estimated from bounding box geometry and projected to GPS coordinates using a spherical earth forward projection.

---

# Decision Logic

```
FOR each image in batch:
    Run YOLOv8 detection
    FOR each bounding box:
        Crop detection region
        Run BioCLIP classification → top species label + score
        Estimate distance from bounding box height + HFOV
        Compute horizontal angle from bounding box center offset
        Project animal GPS from camera GPS + bearing + distance
        Append row to CSV output

After scan:
    Delete local received_images/*.jpg
    Send CLEAR command to ESP32
    Wait for CLEARED acknowledgement
```

---

# Outputs

All outputs are stored locally on the Raspberry Pi SD card.

---

## CSV Detection Log

Saved to:

```
./results/detections_<YYYYMMDD_HHMMSS>.csv
```

---

# CSV Output Schema

One row per confirmed detection.

| Column | Type | Description |
|--------|------|-------------|
| image | String | Source image filename |
| image_path | String | Full path to image |
| timestamp_file | ISO datetime | File modification time |
| timestamp_from_name | ISO datetime | Parsed from filename if available |
| detection_index | Int | Detection index within image |
| bioclip_label | String | Top predicted species |
| bioclip_score | Float | BioCLIP confidence score |
| yolo_conf | Float | YOLO detection confidence |
| xmin / ymin / xmax / ymax | Float | Bounding box coordinates |
| distance_m | Float | Estimated distance to animal (metres) |
| angle_deg | Float | Horizontal angle from camera axis |
| camera_lat | Float | Camera GPS latitude |
| camera_lon | Float | Camera GPS longitude |
| camera_bearing_deg | Float | Camera facing direction (degrees from North) |
| animal_lat | Float | Projected animal latitude |
| animal_lon | Float | Projected animal longitude |
| animal_absolute_bearing_deg | Float | Absolute bearing to animal |

---

# Full Pipeline Example

```bash
# Start receiver + scheduler on Raspberry Pi
python ESP_Reader.py

# Run scan manually on a directory
python scan.py \
  --weights yolov8s_openvino_model \
  --images ./received_images \
  --hfov-deg 42.0 \
  --target-height-m 0.9 \
  --camera-lat 39.835373 \
  --camera-lon -81.738422 \
  --camera-bearing-deg 0 \
  --csv-out ./results/detections.csv \
  --top-k 3
```

---

# GPS Projection Model

Distance estimation uses the **pinhole camera model**:

```
focal_length = (image_width / 2) / tan(HFOV / 2)
distance_m   = (real_height_m × focal_length) / bounding_box_height_px
```

Horizontal angle estimation:

```
dx        = bbox_center_x − image_center_x
angle_deg = atan(dx / focal_length)
```

GPS forward projection uses a **spherical earth model** (Vincenty-style):

```python
lat2 = asin(sin(lat1) × cos(d/R) + cos(lat1) × sin(d/R) × cos(bearing))
lon2 = lon1 + atan2(sin(bearing) × sin(d/R) × cos(lat1),
                    cos(d/R) − sin(lat1) × sin(lat2))
```

### Projection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hfov-deg` | 42.0° | Trail Camera G600 horizontal FOV |
| `--target-height-m` | 0.9 m | Approximate shoulder height of target species |
| `--camera-bearing-deg` | 0.0 | Camera facing direction (clockwise from North) |

> [!NOTE]
> GPS accuracy depends on HFOV calibration and assumed target height. Adjust `--target-height-m` per target species for best results.

---

# BioCLIP Species Classification

WildSense uses **BioCLIP 1** (`hf-hub:imageomics/bioclip`) via `pybioclip`'s `CustomLabelsClassifier`.

Each YOLO-detected crop is classified directly from a PIL Image — no temp file required:

```python
classifier = CustomLabelsClassifier(
    cls_ary=SPECIES_LIST,
    device="cpu",
    model_str="hf-hub:imageomics/bioclip",
)
predictions = classifier.predict(crop)   # PIL Image passed directly
top_label   = predictions[0]["classification"]
top_score   = predictions[0]["score"]
```

Multiple runner-up predictions are optionally recorded as additional CSV columns (`bioclip_label_2`, `bioclip_score_2`, etc.) when `--top-k` > 1.

---

# Serial Communication Protocol

The ESP32-CAM and Raspberry Pi communicate over UART at **115200 baud**.

### ESP32 → Raspberry Pi

| Message | Direction | Description |
|---------|-----------|-------------|
| `READY\n` | ESP32 → Pi | Boot complete |
| `CAPTURING\n` | ESP32 → Pi | Photo triggered |
| `IMG:<path>:<size>\n` | ESP32 → Pi | Image header |
| `[raw JPEG bytes]` | ESP32 → Pi | Image data |
| `END\n` | ESP32 → Pi | Image footer |
| `SD_SAVED:<path>\n` | ESP32 → Pi | Confirm SD write |
| `COOLDOWN:<s>s\n` | ESP32 → Pi | Within cooldown window |
| `CAM_FAIL\n` | ESP32 → Pi | Camera init error |
| `SD_FAIL\n` | ESP32 → Pi | SD card error |
| `CLEARED\n` | ESP32 → Pi | SD wipe complete |

### Raspberry Pi → ESP32

| Message | Direction | Description |
|---------|-----------|-------------|
| `CLEAR\n` | Pi → ESP32 | Delete all JPGs from SD |

---

# Repository Structure

```
MakeOHI-O-2026/
│
├── src/
│   ├── esp32_sensing.ino          # ESP32-CAM firmware (Arduino)
│   ├── ESP_Reader.py              # Raspberry Pi receiver + scheduler
│   └── depth_estimate.py         # YOLO + BioCLIP + GPS scan pipeline
│
├── received_images/               # Incoming images from ESP32 (auto-created)
│
├── results/                       # CSV detection outputs (auto-created)
│   └── detections_<timestamp>.csv
│
└── README.md
```

---

# Success Metrics

## Detection

- Confirmed animal present in YOLO bounding box
- BioCLIP species confidence score recorded per detection

## GPS Projection

- Animal latitude/longitude estimated per detection
- Absolute bearing computed from camera orientation + horizontal offset

## Pipeline Reliability

- Handles serial transmission errors gracefully
- 10-hour batch cycle with automatic SD cleanup
- Cooldown prevents redundant captures per event

---

# Non-Goals

- No cloud infrastructure
- No real-time streaming
- No audio classification
- No multi-camera synchronization
- No ecological forecasting
- No medical or behavioral diagnosis

---

# Long-Term Vision

Future expansions may include:

- Remote cloud upload via cellular module
- Multi-species BioCLIP expansion (full Tree-of-Life)
- Nighttime infrared capture support
- Audio bioacoustic event detection
- Federated edge learning across multiple nodes
- Integration with SmartWilds dataset pipelines

---

# Why WildSense Matters

WildSense demonstrates:

- Deployable edge AI in remote field environments
- Low-cost conservation monitoring with commodity hardware
- Automated species identification without cloud dependency
- Structured, research-ready GPS-tagged datasets from raw camera trap images

The innovation is not just detection —  
it is **automated, deployable, location-aware AI for ecological impact**.
