// Standalone FSR test — no motor, just prints raw ADC and voltage
// Wiring: 3.3V -> FSR -> GPIO34 -> 47k resistor -> GND

#define FSR_PIN 34

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("FSR test starting...");
  Serial.println("Squeeze the sensor and watch the values change.");
}

void loop() {
  int raw = analogRead(FSR_PIN);          // 0-4095 on ESP32 (12-bit ADC)
  float voltage = raw * (3.3 / 4095.0);

  Serial.print("Raw: ");
  Serial.print(raw);
  Serial.print("   Voltage: ");
  Serial.print(voltage, 2);
  Serial.println(" V");

  delay(1000);
}
