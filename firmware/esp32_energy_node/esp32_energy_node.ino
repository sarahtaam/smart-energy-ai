/*
 * Smart Energy AI - ESP32 energy monitoring node
 *
 * Hardware
 *   - ESP32 DevKit v1
 *   - PZEM-004T v3.0 (100A, with split-core CT): voltage, current, power, energy, frequency, PF
 *   - DHT22: ambient temperature / humidity
 *
 * Wiring
 *   PZEM-004T  TX  -> ESP32 GPIO16 (RX2)
 *   PZEM-004T  RX  -> ESP32 GPIO17 (TX2)
 *   PZEM-004T  5V  -> ESP32 VIN (5V)     GND -> GND
 *   DHT22      DATA-> ESP32 GPIO4 (10k pull-up to 3.3V)
 *   DHT22      VCC -> 3.3V               GND -> GND
 *
 *   !! The PZEM AC side is connected to MAINS (230V). Work with the power OFF and
 *   !! have the AC wiring checked by a qualified person.
 *
 * Libraries (Arduino Library Manager)
 *   - PZEM004Tv30      by Jakub Mandula
 *   - PubSubClient     by Nick O'Leary
 *   - ArduinoJson      by Benoit Blanchon (v7)
 *   - DHT sensor library by Adafruit (+ Adafruit Unified Sensor)
 *
 * Publishes JSON to:  smartenergy/<DEVICE_ID>/telemetry
 *   {"device_id":"lab_a","ts":1759480000,"voltage":231.2,"current":2.31,"power":480.5,
 *    "energy":123.456,"frequency":50.0,"pf":0.9,"temperature":24.5,"humidity":55.0}
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <PZEM004Tv30.h>
#include <DHT.h>
#include <time.h>

#include "config.h"  // copy config.h.example -> config.h and edit

#define PZEM_RX_PIN 16
#define PZEM_TX_PIN 17
#define DHT_PIN 4
#define DHT_TYPE DHT22

PZEM004Tv30 pzem(Serial2, PZEM_RX_PIN, PZEM_TX_PIN);
DHT dht(DHT_PIN, DHT_TYPE);
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

char topic[96];
char statusTopic[96];
unsigned long lastPublish = 0;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to WiFi %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi OK, IP=%s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi failed, restarting");
    ESP.restart();
  }
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.printf("Connecting to MQTT %s:%d ... ", MQTT_HOST, MQTT_PORT);
    // Last-will message so the backend knows when the node goes offline
    bool ok = (strlen(MQTT_USER) > 0)
      ? mqtt.connect(DEVICE_ID, MQTT_USER, MQTT_PASSWORD, statusTopic, 1, true, "offline")
      : mqtt.connect(DEVICE_ID, statusTopic, 1, true, "offline");
    if (ok) {
      Serial.println("connected");
      mqtt.publish(statusTopic, "online", true);
    } else {
      Serial.printf("failed rc=%d, retry in 5s\n", mqtt.state());
      delay(5000);
    }
  }
}

// Returns NAN-safe value for JSON (null when the sensor read failed)
void putReading(JsonDocument &doc, const char *key, float value, int decimals) {
  if (isnan(value)) {
    doc[key] = nullptr;
  } else {
    float p = powf(10, decimals);
    doc[key] = roundf(value * p) / p;
  }
}

void publishReading() {
  float voltage = pzem.voltage();
  float current = pzem.current();
  float power = pzem.power();
  float energy = pzem.energy();  // kWh, cumulative (stored in PZEM)
  float frequency = pzem.frequency();
  float pf = pzem.pf();
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // PZEM returns NAN on all fields when there is no mains voltage (outage).
  // Report an outage explicitly as zeros so the backend can detect it.
  if (isnan(voltage)) {
    voltage = 0; current = 0; power = 0; frequency = 0; pf = 0;
  }

  JsonDocument doc;
  doc["device_id"] = DEVICE_ID;
  time_t now = time(nullptr);
  if (now > 1700000000) doc["ts"] = (long)now;  // only send ts once NTP is synced
  putReading(doc, "voltage", voltage, 1);
  putReading(doc, "current", current, 3);
  putReading(doc, "power", power, 1);
  putReading(doc, "energy", energy, 3);
  putReading(doc, "frequency", frequency, 2);
  putReading(doc, "pf", pf, 2);
  putReading(doc, "temperature", temperature, 1);
  putReading(doc, "humidity", humidity, 1);

  char payload[320];
  size_t n = serializeJson(doc, payload, sizeof(payload));
  bool ok = mqtt.publish(topic, (const uint8_t *)payload, n, false);
  Serial.printf("%s %s\n", ok ? "PUB" : "PUB FAILED", payload);
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  snprintf(topic, sizeof(topic), "%s/%s/telemetry", TOPIC_PREFIX, DEVICE_ID);
  snprintf(statusTopic, sizeof(statusTopic), "%s/%s/status", TOPIC_PREFIX, DEVICE_ID);

  connectWiFi();
  configTime(0, 0, "pool.ntp.org", "time.google.com");  // UTC timestamps

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setBufferSize(512);
}

void loop() {
  connectWiFi();
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  if (millis() - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = millis();
    publishReading();
  }
}
