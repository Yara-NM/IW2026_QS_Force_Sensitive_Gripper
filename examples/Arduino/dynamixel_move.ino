#include <AX12A.h>


#define DirectionPin  (21u)
#define BaudRate    (1000000ul)
#define ID    (1u)



void setup()
{
   delay(1000); // Give time for Dynamixel to start on power-up 
   ax12a.begin(BaudRate, DirectionPin, &Serial2);
   ax12a.setEndless(ID, OFF);
   Serial.begin(115200); 
}

void loop()
{ 
    ax12a.ledStatus(ID, ON);
    delay(50);
    ax12a.moveSpeed(ID,0,400);
    Serial.println("on");

    delay(2000);

  

    ax12a.ledStatus(ID, OFF);
    delay(50);
    ax12a.moveSpeed(ID, 131,400);

    delay(2000);

}
