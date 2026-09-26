#pragma once

#include <Arduino.h>

// Mic: the Sense board's PDM microphone → 16 kHz mono s16le frames.
//
// A capture task reads the I2S peripheral in 30 ms chunks and drops each one
// into a queue; loop() drains the queue and ships frames to the brain
// (the WebSocket client isn't safe to call from another task).

class Mic {
 public:
  static const size_t FRAME_SAMPLES = 480;  // 30 ms at 16 kHz
  static const size_t FRAME_BYTES = FRAME_SAMPLES * 2;

  void begin(uint8_t clkPin, uint8_t dataPin, float gain);
  void setStreaming(bool on) { streaming_ = on; }
  bool streaming() const { return streaming_; }

  // Pops one captured frame into `out` (FRAME_BYTES). Returns false if none.
  bool nextFrame(uint8_t* out);

  // RMS (0..1) of the most recent frame, for level checks over serial.
  float level() const { return level_; }

 private:
  static void taskEntry(void* self);
  void task();

  QueueHandle_t queue_ = nullptr;
  volatile bool streaming_ = false;
  volatile float level_ = 0.0f;
  float gain_ = 1.0f;
};
