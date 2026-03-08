# MakeOHI-O-2026
# WildSense

**AI-Powered Wildlife Camera Trap System**

PIR Motion Detection • YOLO Animal Detection • BioCLIP Species Identification • GPS Animal Location

WildSense is an embedded wildlife monitoring system that automatically detects animals, identifies species, and estimates their GPS location using computer vision and geometric projection. The system transforms traditional camera traps into intelligent ecological sensors capable of generating structured datasets for conservation research.

Developed for **Make I/O 2026**.

---

## Overview

Traditional trail cameras capture thousands of images but provide **no automatic species identification or location data**, requiring researchers to manually review images.

WildSense automates this process by combining:

- Motion-triggered camera traps
- AI-based animal detection and classification
- GPS location estimation

The system outputs structured datasets containing **species predictions, confidence scores, and estimated animal coordinates**.

---

## Inspiration

WildSense was inspired by the **SmartWilds dataset**, a multimodal wildlife monitoring project conducted in **Summer 2025 at The Wilds safari park in Ohio**.

SmartWilds collected synchronized:

- Drone imagery  
- Camera trap images and videos  
- Bioacoustic recordings  

During this project we observed how much **manual processing is required to analyze camera trap data**, which motivated the creation of WildSense to automate species identification and spatial analysis.

---

## System Architecture

WildSense uses a **two-device architecture**:

### Camera Node (ESP32-CAM)

Captures wildlife images in the field.

Features:
- ESP32-CAM module
- PIR motion sensor
- 1600×1200 JPEG capture
- MicroSD storage
- 1-minute capture cooldown
- UART transmission to Raspberry Pi

Images are transmitted using a custom serial protocol:
