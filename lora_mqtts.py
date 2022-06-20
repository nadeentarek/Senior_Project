
import paho.mqtt.client as mqtt #import the client1
import json,base64
from lora.crypto import loramac_decrypt
def on_connect(client, userdata, flags, rc):


    print("Connected with result code "+str(rc))
    client.subscribe("#")

def on_message(client, userdata, msg):
    #print("received message =",str(msg.payload.decode("utf-8")))
    print(msg.topic)
    if 'up' in msg.topic:
    	a = str(msg.payload.decode('utf-8'))
    	#print(str(msg.payload.phyPayload.decode('utf-8')))
    	y=json.loads(a)
    	p = y['phyPayload']
    	#counter = 2
    ##	session_key = '2636ee7d406ae6484fe89f7936abe883'
    #	device_add = '000af974'
    #	f = loramac_decrypt(p,counter,session_key,device_add)
    #	print(f)
#b = base64.b64decode(y['phyPayload'].encode())
    	print('decoded message ',a)
    	#print('json encoded ',y)
    	#print('b typee ',type(b))
    	#print('encoded thingy ',b.decode())
    	public_client.publish("siloAnomaly/1", msg.payload)

broker_address="192.168.1.126"

public_broker_address="broker.hivemq.com" #use external broker

client = mqtt.Client() #create new instance
public_client = mqtt.Client()



def main():
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker_address, 1883, 60)
    public_client.connect(public_broker_address, 1883, 60000)
    client.loop_forever()

#    public_client.loop_forever()
    
if __name__ == '__main__':
    main()
