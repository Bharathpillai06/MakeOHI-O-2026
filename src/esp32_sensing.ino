#include "esp_camera.h"
#include "Arduino.h"
#include "FS.h"
#include "SD_MMC.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "driver/rtc_io.h"
#include <EEPROM.h>

// ── Pin definitions (AI Thinker ESP32-CAM) ───────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ── WildSense pins ────────────────────────────────────────────────────────
const int pirPin = 13;
const int ledPin = 2;

// ── Cooldown ──────────────────────────────────────────────────────────────
const unsigned long COOLDOWN_MS = 60000;   // 1 minute
unsigned long lastPhotoTime     = -60000UL;

// ── EEPROM ────────────────────────────────────────────────────────────────
#define EEPROM_SIZE 1
int pictureNumber = 0;

// ── Forward declarations ──────────────────────────────────────────────────
void takePhotoAndSend();
void clearSDImages();

// ─────────────────────────────────────────────────────────────────────────
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);   // communicates with Raspberry Pi
  pinMode(pirPin, INPUT);
  pinMode(ledPin, OUTPUT);

  // ── Camera config ──────────────────────────────────────────────────────
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size   = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("CAM_FAIL:0x%x\n", err);
    return;
  }

  if (!SD_MMC.begin()) {
    Serial.println("SD_FAIL");
    return;
  }
  if (SD_MMC.cardType() == CARD_NONE) {
    Serial.println("SD_FAIL:NO_CARD");
    return;
  }

  EEPROM.begin(EEPROM_SIZE);
  pictureNumber = EEPROM.read(0);

  Serial.println("READY");
}

// ─────────────────────────────────────────────────────────────────────────
void loop() {
  // ── Check for commands from Pi ─────────────────────────────────────────
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "CLEAR") {
      Serial.println("Clearing SD...");
      clearSDImages();
      Serial.println("CLEARED");
    }
  }

  // ── PIR motion detection ───────────────────────────────────────────────
  int pirState = digitalRead(pirPin);

  if (pirState == HIGH) {
    digitalWrite(ledPin, HIGH);

    unsigned long now = millis();
    if (now - lastPhotoTime >= COOLDOWN_MS) {
      takePhotoAndSend();
      lastPhotoTime = now;
    } else {
      unsigned long remaining = (COOLDOWN_MS - (now - lastPhotoTime)) / 1000;
      Serial.printf("COOLDOWN:%lus\n", remaining);
    }

  } else {
    digitalWrite(ledPin, LOW);
  }

  delay(100);
}

// ─────────────────────────────────────────────────────────────────────────
void takePhotoAndSend() {
  Serial.println("CAPTURING");

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("CAM_FAIL");
    return;
  }

  pictureNumber++;
  String path = "/picture" + String(pictureNumber) + ".jpg";

  // ── Save to SD ─────────────────────────────────────────────────────────
  fs::FS &fs = SD_MMC;
  File file  = fs.open(path.c_str(), FILE_WRITE);
  if (file) {
    file.write(fb->buf, fb->len);
    file.close();
    EEPROM.write(0, pictureNumber);
    EEPROM.commit();
    Serial.printf("SD_SAVED:%s\n", path.c_str());
  } else {
    Serial.println("SD_WRITE_FAIL");
  }

  // ── Send to Pi over Serial ─────────────────────────────────────────────
  // Protocol:
  //   1. Header:  "IMG:<filename>:<size>\n"
  //   2. Raw JPEG bytes
  //   3. Footer:  "END\n"
  Serial.printf("IMG:%s:%d\n", path.c_str(), fb->len);
  Serial.write(fb->buf, fb->len);
  Serial.print("END\n");

  esp_camera_fb_return(fb);
}

// ─────────────────────────────────────────────────────────────────────────
void clearSDImages() {
  fs::FS &fs = SD_MMC;
  File root  = fs.open("/");
  if (!root || !root.isDirectory()) {
    Serial.println("SD_FAIL:CANT_OPEN_ROOT");
    return;
  }

  int deleted = 0;
  File file   = root.openNextFile();
  while (file) {
    String name = "/" + String(file.name());
    bool isJpg  = name.endsWith(".jpg") || name.endsWith(".JPG");
    file.close();

    if (isJpg) {
      if (fs.remove(name)) {
        deleted++;
      }
    }
    file = root.openNextFile();
  }

  // Reset picture counter
  pictureNumber = 0;
  EEPROM.write(0, 0);
  EEPROM.commit();

  Serial.printf("Deleted %d images from SD.\n", deleted);
}
