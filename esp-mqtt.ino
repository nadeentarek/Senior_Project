/*
 *  This sketch demonstrates how to scan WiFi networks.
 *  The API is almost the same as with the WiFi Shield library,
 *  the most obvious difference being the different file you need to include:
 */
#include <Arduino.h>
#include "WiFi.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
//#include <TimeLib.h>     // for update/display of time
#include "time.h"

//1
/*double   myNumbers[] = {19.91      , 19.90911111, 19.90822222, 19.90733333, 19.90644444,
       19.90555556, 19.90466667, 19.90377778, 19.90288889, 19.902     ,
       19.90111111, 19.90022222, 19.89933333, 19.89844444, 19.89755556,
       19.89666667, 19.89577778, 19.89488889, 19.894     , 19.89311111,
       19.89222222, 19.89133333, 19.89044444, 19.88955556, 19.88866667, 22.484     , 22.48477778, 22.48555556, 22.48633333, 22.48711111,
       22.48788889, 22.48866667, 22.48944444, 22.49022222, 22.491     ,
       22.49177778, 22.49255556, 22.49333333, 22.49411111, 22.49488889,
       22.49566667, 22.49644444, 22.49722222, 22.498     , 22.49877778,
       22.49955556, 22.50033333, 22.50111111, 22.50188889, 22.50266667};
*/

//2
/*double   myNumbers[] = {18.82533333, 18.826     , 18.82666667, 18.82733333, 18.828     ,
       18.82866667, 18.82933333, 18.83      , 18.83      , 18.83      ,
       18.83      , 18.83      , 18.83      , 18.83      , 18.83      ,
       18.83      , 18.83      , 18.83      , 18.83      , 18.83      ,
       18.83      , 18.83      , 18.83      , 18.83066667, 18.83133333, 41.072, 41.074, 41.076, 41.078, 41.08 , 41.082, 41.084, 41.086,
       41.088, 41.09 , 41.092, 41.094, 41.096, 41.098, 41.1  , 41.102,
       41.104, 41.106, 41.108, 41.11 , 41.112, 41.114, 41.116, 41.118,
       41.12 };*/
//2   
double   myNumbers[] = {18.82533333, 18.826     , 18.82666667, 18.82733333, 41.072, 41.074, 41.076, 41.078}; 

//1
//double   myNumbers[] = {19.91      , 19.90911111, 19.90822222, 19.90733333, 19.90644444, 22.49955556, 22.50033333, 22.50111111, 22.50188889, 22.50266667};

// NTP server to request epoch time
const char* ntpServer = "pool.ntp.org";

// Variable to save current epoch time
unsigned long epochTime;


// MQTT client
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient); 
const char *mqttServer = "192.168.43.164";
//const char *mqttServer = "broker.emqx.io";
int mqttPort = 1883;

//connect to the wifi
const char *SSID = "N00b";
const char *PWD = "32456xxy";

void connectToWiFi() {
  Serial.print("Connectiog to ");
 
  WiFi.begin(SSID, PWD);
  Serial.println(SSID);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.print("Connected.");
  
}
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Callback - ");
  Serial.print("Message:");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
}
void setupMQTT() {
  mqttClient.setServer(mqttServer, mqttPort);
  // set the callback function
  mqttClient.setCallback(callback);
}


void reconnect() {
  Serial.println("Connecting to MQTT Broker...");
  while (!mqttClient.connected()) {
      Serial.println("Reconnecting to MQTT Broker..");
      String clientId = "ESP32Client-";
      clientId += String(random(0xffff), HEX);
      
      if (mqttClient.connect(clientId.c_str())) {
        Serial.println("Connected.");
        // subscribe to topic
        mqttClient.subscribe("/swa/commands");
      }
      
  }
}


// Function that gets current epoch time
unsigned long getTime() {
  time_t now;
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    //Serial.println("Failed to obtain time");
    return(0);
  }
  time(&now);
  return now;
}
void setup()
{
    Serial.begin(115200);
    connectToWiFi();
    setupMQTT();
    configTime(0, 0, ntpServer);
    delay(100);

}

void loop()
{
  
  if (!mqttClient.connected())
    reconnect();
    mqttClient.loop();
  
  //Json
  for (int i=0; i < 8; i++){
    delay(10000);
    StaticJsonBuffer<300> JSONbuffer;
    JsonObject& JSONencoder = JSONbuffer.createObject();
    epochTime = getTime();
    JSONencoder["time"] = epochTime ;
    JSONencoder["sensorId"] = "2a";                                     //1a, 2a
    JSONencoder["sensorReading"] = myNumbers[i];
  
    char JSONmessageBuffer[100];
    JSONencoder.printTo(JSONmessageBuffer, sizeof(JSONmessageBuffer));
    Serial.println("Sending message to MQTT topic..");
    Serial.println(JSONmessageBuffer);
    
     if (mqttClient.publish("silo1Temp/2", JSONmessageBuffer) == true) { //1,2
      Serial.println("Success sending message");
    } else {
      Serial.println("Error sending message");
    }

  }
  //mqttClient.publish("temp/1", "hi from sensor 1");

}
