// Bluetooth test v2 — fixes the empty-echo flood.
// The buffer now persists across loop() calls, so it correctly
// accumulates characters even when they arrive one at a time
// (e.g. typed live over a raw terminal like screen/picocom).
// It only echoes once a full line (ending in '\n') has arrived.

#include <BluetoothSerial.h>

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` and enable it
#endif

#define BT_NAME "GRIPPER_TEST"

BluetoothSerial SerialBT;
String btBuffer = "";  // persists across loop() calls

void setup() {
  Serial.begin(115200);
  SerialBT.begin(BT_NAME);
  Serial.println("Bluetooth test starting...");
  Serial.print("Device name: ");
  Serial.println(BT_NAME);
  Serial.println("Pair with this device from your PC, then send some text.");
}

void loop() {
  while (SerialBT.available()) {
    char c = SerialBT.read();

    if (c == '\n' || c == '\r') {
      if (btBuffer.length() > 0) {
        Serial.print("Received over BT: ");
        Serial.println(btBuffer);

        SerialBT.print("Echo: ");
        SerialBT.println(btBuffer);

        btBuffer = "";
      }
      // ignore \r or repeated \n with nothing typed yet
    } else {
      btBuffer += c;
    }
  }

  // Data typed into USB Serial Monitor -> send it out over Bluetooth
  if (Serial.available()) {
    String toSend = Serial.readStringUntil('\n');
    SerialBT.println(toSend);
    Serial.print("Sent over BT: ");
    Serial.println(toSend);
  }
}