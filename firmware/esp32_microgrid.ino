/*
 * =====================================================================================
 *  SELF-LEARNING MICROGRID ENERGY OPTIMIZATION & BATTERY HEALTH PREDICTION SYSTEM
 * =====================================================================================
 *  Target Hardware: ESP32 Dev Module
 *  Pins:
 *    - Pin D35 : Battery Voltage / SoC Divider (ADC)
 *    - Pin 34  : Solar Voltage Divider (ADC)
 *    - Pin 32  : ACS712 Current Sensor (ADC)
 *    - Pin 23  : Servo Motor Signal (Solar tracking / Load router)
 *    - Pin 27  : Status LED Indicator
 *    - Pin 26  : Relay Module (Load Switching ON/OFF)
 * =====================================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// WiFi Configuration
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Backend Server URL (IP of the computer running the Digital Twin FastAPI backend)
const char* serverUrl = "http://192.168.1.100:8000"; // Replace with your laptop's LAN IP

// Pin Definitions
const int PIN_BATTERY_ADC = 35; // Pin D35 Battery level
const int PIN_SOLAR_ADC   = 34; // Solar voltage
const int PIN_ACS712      = 32; // ACS712 Current sensor
const int PIN_SERVO       = 23; // Pin 23 Servo
const int PIN_LED         = 27; // Pin 27 LED
const int PIN_RELAY       = 26; // Load Relay

Servo solarServo;

// System State
String currentMode = "BATTERY"; // "SOLAR", "BATTERY", "AUTO"
bool loadStatus = true;         // true = ON, false = OFF
int currentServoAngle = 0;
bool ledStatus = false;

// Calibration Constants
const float VOLTAGE_DIVIDER_RATIO = 5.0; // Adjust for your resistor divider
const float ACS712_SENSITIVITY = 0.185;  // 185mV/A for 5A ACS712

float readSolarVoltage() {
  int raw = analogRead(PIN_SOLAR_ADC);
  float vOut = (raw / 4095.0) * 3.3;
  return vOut * VOLTAGE_DIVIDER_RATIO;
}

float readBatteryVoltage() {
  int raw = analogRead(PIN_BATTERY_ADC);
  float vOut = (raw / 4095.0) * 3.3;
  return vOut * VOLTAGE_DIVIDER_RATIO;
}

float readBatteryPercentage(float voltage) {
  // Approximate for 4S Li-Ion or 12V Lead Acid
  float soc = ((voltage - 11.5) / (16.8 - 11.5)) * 100.0;
  return constrain(soc, 0.0, 100.0);
}

float readLoadCurrent() {
  int raw = analogRead(PIN_ACS712);
  float voltage = (raw / 4095.0) * 3.3;
  float current = abs((voltage - 1.65) / ACS712_SENSITIVITY);
  return (current < 0.05) ? 0.0 : current; // Noise filter
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, HIGH); // Default Load ON

  solarServo.attach(PIN_SERVO);
  solarServo.write(0);

  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("Device IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    float solarV = readSolarVoltage();
    float batV = readBatteryVoltage();
    float batLevel = readBatteryPercentage(batV);
    float loadI = readLoadCurrent();
    float loadW = batV * loadI;

    // 1. Send Telemetry to Digital Twin Backend
    HTTPClient http;
    String telemetryUrl = String(serverUrl) + "/telemetry";
    http.begin(telemetryUrl);
    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<384> doc;
    doc["device_id"] = "ESP32-MICROGRID-001";
    doc["solar_voltage"] = solarV;
    doc["battery_voltage"] = batV;
    doc["battery_level"] = batLevel;
    doc["load_current"] = loadI;
    doc["load_power"] = loadW;
    doc["mode"] = currentMode;
    doc["load_status"] = loadStatus ? "ON" : "OFF";
    doc["servo_angle"] = currentServoAngle;
    doc["led_status"] = ledStatus ? "ON" : "OFF";

    String payload;
    serializeJson(doc, payload);
    int httpCode = http.POST(payload);
    http.end();

    // 2. Poll Digital Twin for Remote Control Commands
    String controlUrl = String(serverUrl) + "/control/state";
    http.begin(controlUrl);
    int getCode = http.GET();
    if (getCode == 200) {
      String response = http.getString();
      StaticJsonDocument<256> ctrlDoc;
      deserializeJson(ctrlDoc, response);

      currentMode = ctrlDoc["mode"].as<String>();
      String newLoad = ctrlDoc["load_status"].as<String>();
      loadStatus = (newLoad == "ON");
      currentServoAngle = ctrlDoc["servo_angle"].as<int>();
      ledStatus = (ctrlDoc["led_status"].as<String>() == "ON");

      // Apply Actuation
      digitalWrite(PIN_RELAY, loadStatus ? HIGH : LOW);
      digitalWrite(PIN_LED, ledStatus ? HIGH : LOW);
      solarServo.write(currentServoAngle);
    }
    http.end();
  }

  delay(2000); // Send every 2 seconds
}
