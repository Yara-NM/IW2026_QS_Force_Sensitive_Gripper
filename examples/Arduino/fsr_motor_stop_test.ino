// Combined test: motor cycles open/close, but stops moving if the
// FSR reading crosses FORCE_THRESHOLD. Tune the threshold using the
// raw values you saw from fsr_test.ino before relying on this.

#include <AX12A.h>

#define DirectionPin (21u)
#define BaudRate (1000000ul)
#define ID (1u)
#define FSR_PIN 34
#define FORCE_THRESHOLD 2000   // <-- placeholder, replace with a real value from fsr_test.ino

void setup() {
  delay(1000); // give the Dynamixel time to boot
  ax12a.begin(BaudRate, DirectionPin, &Serial2);
  ax12a.setEndless(ID, OFF);
  Serial.begin(115200);
}

void loop() {
  int force = analogRead(FSR_PIN);
  Serial.print("Force: ");
  Serial.println(force);

  if (force < FORCE_THRESHOLD) {
    // no significant press yet -> keep cycling
    ax12a.ledStatus(ID, ON);
    ax12a.moveSpeed(ID, 0, 400);
    delay(2000);

    ax12a.ledStatus(ID, OFF);
    ax12a.moveSpeed(ID, 131, 400);
    delay(2000);
  } else {
    // force threshold exceeded -> stop moving
    ax12a.moveSpeed(ID, 0, 0); // speed 0 halts the motor
    Serial.println("Force threshold exceeded -- stopping");
    delay(200);
  }
}
