# MakeOHI-O-2026
# WildSense

**AI-Powered Wildlife Camera Trap System**

PIR Motion Detection • YOLO Animal Detection • BioCLIP Species Identification • GPS Animal Location

WildSense is an embedded wildlife monitoring system that automatically detects animals, identifies species, and estimates their GPS location using computer vision and geometric projection. The system transforms traditional camera traps into intelligent ecological sensors capable of generating structured datasets for conservation research.

Developed for **Make I/O 2026**.

---

## Overview

Traditional trail cameras capture thousands of images but provide **no automatic species identification or location data**, requiring researchers to manually review images.

WildSense automates this process by combining motion-triggered camera traps, AI-based animal detection and classification, and GPS location estimation. The system outputs structured datasets containing **species predictions, confidence scores, and estimated animal coordinates**.

---

## Inspiration

WildSense was inspired by the **SmartWilds dataset** Bharath collected, a multimodal wildlife monitoring project conducted in **Summer 2025 at The Wilds safari park in Ohio**.

[SmartWilds] (https://arxiv.org/pdf/2509.18894)

SmartWilds collected synchronized:

- Drone imagery  
- Camera trap photos and videos  
- Bioacoustic recordings  

During this project we observed how much **manual processing researchers must perform to analyze camera trap data**, which motivated the creation of WildSense to automate species identification and spatial analysis.

---

## System Architecture

WildSense uses a **two-device architecture** consisting of a camera node and a processing node.

### Camera Node (ESP32-CAM)

The camera node captures wildlife images in the field.

Features:

- ESP32-CAM module  
- PIR motion sensor  
- 1600×1200 JPEG capture  
- MicroSD storage  
- 1-minute capture cooldown  
- UART serial transmission  

Images are transmitted using a custom serial protocol:

```
IMG → raw image bytes → END
```

The ESP32 listens for a `CLEAR` command from the processing unit to erase images after they are processed.

---

### Processing Node (Raspberry Pi)

The Raspberry Pi receives images and performs the AI processing pipeline.

Responsibilities include:

- Receiving images over UART  
- Timestamping and storing images  
- Running computer vision inference  
- Estimating animal GPS position  
- Exporting structured detection datasets  

The system runs a threaded pipeline consisting of a **UART image receiver and a scheduled inference scan**.

---

## AI Pipeline

WildSense uses a **two-stage computer vision pipeline**.

### Stage 1 — YOLOv8

YOLOv8 detects animals in the image and generates bounding boxes.

- OpenVINO optimized for CPU inference  
- Image size: 640×640  
- Confidence threshold: 0.25  

Each detected bounding box is cropped and passed to the classification stage.

---

### Stage 2 — BioCLIP

BioCLIP is a biology-focused vision-language model used for species identification.

Features:

- Tree-of-Life trained model  
- Custom label list for site-specific species  
- Direct classification from image crops  
- Outputs predicted species and confidence score  

---

## GPS Estimation

WildSense estimates the **location of the detected animal** using camera geometry.

Steps:

1. Compute focal length from camera field-of-view  
2. Estimate distance using bounding box height  
3. Calculate horizontal angle relative to image center  
4. Combine with camera bearing  
5. Project the animal's GPS coordinates from the camera location  

Distance estimation formula:

```
distance = (target_height × focal_length) / bbox_height
```

---

## Output Dataset

Each detection is exported as a row in a CSV dataset.

Example fields include:

```
image
timestamp_file
detection_index
bioclip_label
bioclip_score
yolo_conf
xmin ymin xmax ymax
distance_m
angle_deg
camera_lat camera_lon
animal_lat animal_lon
animal_absolute_bearing_deg
```

---

## Results

Initial deployment produced:

- **28,376 detections**
- **6 camera sites**
- **11 tracked species**

WildSense automatically generated species predictions and estimated GPS coordinates from camera trap imagery.

---

## Hardware Requirements

Camera Node:

- ESP32-CAM  
- PIR motion sensor  
- MicroSD card  
- 5V power supply  

Processing Node:

- Raspberry Pi  
- UART connection  
- Local storage for image datasets  

Recommended for field deployment:

- 18650 battery pack  
- MT3608 boost converter  

---

## Future Work

Planned improvements include:

- Multi-camera monitoring networks  
- Cross-triangulated animal locations  
- Long-range wireless image transfer  
- Solar-powered deployments  
- Expanded species classification models  
- Automated wildlife heatmaps  

---

## Impact

WildSense transforms passive trail cameras into **intelligent ecological monitoring systems**, enabling faster wildlife analysis for:

- Conservation research  
- Biodiversity monitoring  
- Habitat analysis  
- Environmental education  

---

## Authors

**Manas Tripathi**  
**Bharath Pillai**

Make I/O 2026

---

## License

MIT License
