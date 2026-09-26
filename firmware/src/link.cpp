#include "link.h"

#include <ArduinoJson.h>
#include <WiFi.h>

static const char* FW_VERSION = "0.3.0";

void Link::begin(const char* ssid, const char* pass, const char* host,
                 uint16_t port, const char* token, CommandHandler onCommand,
                 StateHandler onState) {
  ssid_ = ssid;
  pass_ = pass;
  token_ = token;
  onCommand_ = std::move(onCommand);
  onState_ = std::move(onState);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // lower latency; we're on wall power anyway
  WiFi.begin(ssid_, pass_);
  Serial.printf("wifi: joining \"%s\"...\n", ssid_);

  ws_.begin(host, port, "/");
  ws_.onEvent([this](WStype_t type, uint8_t* payload, size_t length) {
    onEvent(type, payload, length);
  });
  ws_.setReconnectInterval(3000);
  ws_.enableHeartbeat(15000, 3000, 2);  // ping every 15 s, drop after 2 misses
}

void Link::update(uint32_t nowMs) {
  bool wifiUp = WiFi.status() == WL_CONNECTED;
  if (wifiUp && !wifiReported_) {
    Serial.printf("wifi: connected, ip %s\n", WiFi.localIP().toString().c_str());
    wifiReported_ = true;
  } else if (!wifiUp && wifiReported_) {
    Serial.println(F("wifi: lost, retrying"));
    wifiReported_ = false;
  }
  if (!wifiUp) return;  // the WiFi stack reconnects by itself

  ws_.loop();
}

void Link::sendJson(const String& json) {
  if (!connected_) return;
  ws_.sendTXT(json.c_str(), json.length());
}

void Link::sendBinary(uint8_t type, const uint8_t* data, size_t len) {
  if (!connected_) return;
  static uint8_t* buf = nullptr;
  static const size_t CAP = 64 * 1024;  // mic frames are ~1 KB, JPEGs ~10-20 KB
  if (buf == nullptr) buf = static_cast<uint8_t*>(ps_malloc(CAP));
  if (buf == nullptr || len > CAP - 1) return;
  buf[0] = type;
  memcpy(buf + 1, data, len);
  ws_.sendBIN(buf, len + 1);
}

void Link::sendState(float panDeg, const char* emotion) {
  if (!connected_) return;
  JsonDocument doc;
  doc["type"] = "state";
  doc["pan"] = panDeg;
  doc["emotion"] = emotion;
  String out;
  serializeJson(doc, out);
  sendJson(out);
}

void Link::onEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED: {
      connected_ = true;
      Serial.println(F("brain: connected"));
      JsonDocument doc;
      doc["type"] = "hello";
      doc["who"] = "desk-robot";
      doc["fw"] = FW_VERSION;
      doc["token"] = token_;  // the brain drops connections without it
      String out;
      serializeJson(doc, out);
      sendJson(out);
      if (onState_) onState_(true);
      break;
    }
    case WStype_DISCONNECTED:
      if (connected_) {
        Serial.println(F("brain: disconnected, will retry"));
        if (onState_) onState_(false);
      }
      connected_ = false;
      break;
    case WStype_TEXT: {
      // Copy so we can guarantee a terminator.
      String msg(reinterpret_cast<const char*>(payload), length);
      handleMessage(msg.c_str());
      break;
    }
    case WStype_BIN:
      if (length > 1 && payload[0] == 0x01 && onAudio_) {
        onAudio_(payload + 1, length - 1);
      }
      break;
    default:
      break;
  }
}

void Link::handleMessage(const char* json) {
  JsonDocument doc;
  if (deserializeJson(doc, json)) {
    Serial.printf("brain: bad json: %s\n", json);
    return;
  }
  const char* type = doc["type"] | "";
  String cmd;
  if (!strcmp(type, "emotion")) {
    cmd = String("emo ") + (doc["name"] | "neutral");
  } else if (!strcmp(type, "pan") || !strcmp(type, "tilt")) {
    cmd = String(type) + " " + String(doc["deg"] | 0.0f, 1);
  } else if (!strcmp(type, "speak_begin")) {
    cmd = String("speak_begin ") + String(doc["bytes"] | 0);
  } else if (!strcmp(type, "speak_end")) {
    cmd = type;
  } else if (!strcmp(type, "volume")) {
    cmd = String("volume ") + String(doc["level"] | 0.5f, 2);
  } else if (!strcmp(type, "mic")) {
    cmd = String("mic ") + ((doc["on"] | false) ? "on" : "off");
  } else if (!strcmp(type, "glance")) {
    cmd = String("glance ") + ((doc["on"] | true) ? "on" : "off");
  } else if (!strcmp(type, "asleep")) {
    cmd = String("sleep ") + ((doc["on"] | false) ? "on" : "off");
  } else if (!strcmp(type, "stream")) {
    cmd = String("stream ") + ((doc["on"] | false) ? "on " : "off ") + String(doc["fps"] | 10.0f, 1);
  } else {
    Serial.printf("brain: unknown message type \"%s\"\n", type);
    return;
  }
  if (onCommand_) onCommand_(cmd);
}
