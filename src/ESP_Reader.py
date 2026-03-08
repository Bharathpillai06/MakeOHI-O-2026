#!/usr/bin/env python3
"""
WildSense — Raspberry Pi receiver + YOLO/BioCLIP scanner
- Receives images from ESP32 over serial
- Every 10 hours: runs scan.py on accumulated images, then clears them from SD
"""

import os
import sys
import time
import serial
import datetime
import subprocess
import threading

# ── Config ─────────────────────────────────────────────────────────────────
SERIAL_PORT    = "/dev/ttyS0"
BAUD_RATE      = 115200
IMAGE_DIR      = "./received_images"
RESULTS_DIR    = "./results"
SCAN_SCRIPT    = "./scan.py"
SCAN_INTERVAL  = 10 * 60 * 60      # 10 hours in seconds
YOLO_WEIGHTS   = "yolov8s_openvino_model"
DEVICE         = "cpu"

os.makedirs(IMAGE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
print(f"[WildSense] Listening on {SERIAL_PORT}...")


# ─────────────────────────────────────────────────────────────────────────────
def receive_images():
    """Continuously receive images from ESP32 over serial."""
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line == "READY":
                print("[ESP32] Online and ready.")

            elif line.startswith("IMG:"):
                # Header format: IMG:/picture1.jpg:12345
                try:
                    parts    = line.split(":")
                    filename = parts[1]
                    size     = int(parts[2])
                except Exception as e:
                    print(f"[ERROR] Bad header: {line} — {e}")
                    continue

                print(f"[ESP32] Receiving {filename} ({size} bytes)...")

                # Read image bytes
                image_data = b""
                while len(image_data) < size:
                    chunk = ser.read(size - len(image_data))
                    if not chunk:
                        print("[ERROR] Timeout reading image data")
                        break
                    image_data += chunk

                # Read END footer
                footer = ser.readline().decode("utf-8", errors="ignore").strip()
                if footer != "END":
                    print(f"[WARNING] Expected END, got: {footer}")

                # Save image with timestamp prefix
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{timestamp}{filename}"
                save_path = os.path.join(IMAGE_DIR, save_name)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                print(f"[SAVED] {save_path}")

            elif line.startswith("CAM_FAIL") or line.startswith("SD_FAIL"):
                print(f"[ESP32 ERROR] {line}")

            elif line.startswith("Deleted"):
                print(f"[ESP32] {line}")

        except Exception as e:
            print(f"[SERIAL ERROR] {e}")
            time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
def run_scan():
    """Run scan.py on accumulated images and save CSV."""
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
    if not images:
        print("[SCAN] No images to process — skipping.")
        return

    print(f"[SCAN] Starting scan on {len(images)} images...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out   = os.path.join(RESULTS_DIR, f"detections_{timestamp}.csv")

    cmd = [
        sys.executable, SCAN_SCRIPT,
        "--images",  IMAGE_DIR,
        "--weights", YOLO_WEIGHTS,
        "--csv-out", csv_out,
        "--device",  DEVICE,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout)
        if result.stderr:
            print(f"[SCAN STDERR] {result.stderr}")
        print(f"[SCAN] Complete. CSV saved to: {csv_out}")
    except subprocess.TimeoutExpired:
        print("[SCAN] Timed out after 1 hour.")
        return
    except Exception as e:
        print(f"[SCAN ERROR] {e}")
        return

    clear_local_images()
    clear_esp32_sd()


# ─────────────────────────────────────────────────────────────────────────────
def clear_local_images():
    """Delete all JPGs from local received_images folder."""
    deleted = 0
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith(".jpg"):
            os.remove(os.path.join(IMAGE_DIR, fname))
            deleted += 1
    print(f"[CLEANUP] Deleted {deleted} local images.")


# ─────────────────────────────────────────────────────────────────────────────
def clear_esp32_sd():
    """Send CLEAR command to ESP32 and wait for acknowledgement."""
    print("[CLEANUP] Sending CLEAR command to ESP32...")
    ser.write(b"CLEAR\n")
    for _ in range(30):
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "CLEARED":
            print("[CLEANUP] ESP32 SD card cleared successfully.")
            return
        time.sleep(1)
    print("[WARNING] No CLEAR acknowledgement received from ESP32.")


# ─────────────────────────────────────────────────────────────────────────────
def scan_scheduler():
    """Run scan every 10 hours, starting immediately on boot."""
    while True:
        run_scan()
        print(f"[SCHEDULER] Next scan in {SCAN_INTERVAL // 3600} hours.")
        time.sleep(SCAN_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    receiver_thread  = threading.Thread(target=receive_images,  daemon=True)
    scheduler_thread = threading.Thread(target=scan_scheduler,  daemon=True)

    receiver_thread.start()
    scheduler_thread.start()

    print("[WildSense] Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WildSense] Shutting down.")
