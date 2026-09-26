#pragma once

#include <Arduino.h>
#include <WebSocketsClient.h>
#include <functional>

// Link: WiFi + a WebSocket to the brain server.
//
// Joins WiFi, connects to ws://<host>:<port>, reconnects on its own, and
// turns the server's JSON commands (docs/protocol.md) into the same text
// commands the serial console uses — so "emotion happy" from the brain and
// "emo happy" typed over USB take the exact same path.

class Link {
 public:
  using CommandHandler = std::function<void(const String&)>;
  using StateHandler = std::function<void(bool connected)>;
  using AudioHandler = std::function<void(const uint8_t* pcm, size_t len)>;

  void begin(const char* ssid, const char* pass, const char* host,
             uint16_t port, const char* token, CommandHandler onCommand,
             StateHandler onState = nullptr);

  // Binary frames from the brain: type byte 0x01 = TTS audio (16 kHz s16le).
  void onAudio(AudioHandler h) { onAudio_ = std::move(h); }

  // Call every loop().
  void update(uint32_t nowMs);

  bool connected() const { return connected_; }

  // Robot → server. Text is sent as-is (should be JSON).
  void sendJson(const String& json);
  void sendState(float panDeg, const char* emotion);
  // Binary frame: one type byte (0x01 = mic audio, 0x02 = JPEG) + payload.
  void sendBinary(uint8_t type, const uint8_t* data, size_t len);

 private:
  void onEvent(WStype_t type, uint8_t* payload, size_t length);
  void handleMessage(const char* json);

  WebSocketsClient ws_;
  CommandHandler onCommand_;
  StateHandler onState_;
  AudioHandler onAudio_;
  const char* ssid_ = nullptr;
  const char* pass_ = nullptr;
  const char* token_ = "";
  bool connected_ = false;
  bool wifiReported_ = false;
  uint32_t nextStateMs_ = 0;
};
