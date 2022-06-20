import os, json
from datetime import datetime
from queue import Queue
from threading import Thread
from time import time
#import tensorflow as tf
import paho.mqtt.client as mqtt
from test import test_function
from src.utils import *
from main2 import inference
from datetime import datetime

#from power_predictor_new import predict_power
import numpy as np
#from tflite_runtime.interpreter import Interpreter
s1_recieved = False
s2_recieved = False
testD_main = np.full((11,9,2), 19)
testO_main = np.full((11,2), 19)
s1_temp = 0
s2_temp = 0
sensor_time = 0
s2_id = 0
s1_id = 0
s1_array = np.array([[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]],[[19.91 ],[19.90911111],[19.90822222],[19.90733333],[19.90644444],[19.90555556],[19.90466667],[19.90377778],[19.90288889]]])

s2_array = np.array([[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]],[[18.82533333],[18.826],[18.82666667],[ 18.82733333],[18.828],[18.82866667],[18.82933333],[18.83 ],[18.83 ]]])

#print("Stored array",s1_array.shape)
#print("Stored array",s2_array.shape)

#model_path = "clean_enc.tflite"
# Load model (interpreter)
#interpreter = Interpreter(model_path)
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#print(input_details)
client = mqtt.Client()

queue = Queue()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("silo1Temp/#")

def on_message(client, userdata, msg):
    global s1_recieved
    global s2_recieved
    global testD_main
    global testO_main
    global s2_temp
    global s1_temp
    global s2_id
    global s1_id
    
    #print("got something!")
    #new_input = np.array([[0.068428279,0,0.889573295,0.363636364,0.746389433,0.010989011]])
    print("Received an MQTT message of topic: ",msg.topic)

    if(msg.topic == "silo1Temp/1"):
        s1_recieved = True
        s1_temp = json.loads(msg.payload.decode('utf-8'))["sensorReading"]
        sensor_time = json.loads(msg.payload.decode('utf-8'))["time"]
        #print(sensor_time)
        s1_id = json.loads(msg.payload.decode('utf-8'))["sensorId"]
    elif (msg.topic == "silo1Temp/2"):
        s2_recieved = True
        s2_temp = json.loads(msg.payload.decode('utf-8'))["sensorReading"]
        sensor_time = json.loads(msg.payload.decode('utf-8'))["time"]
        #print(sensor_time)
        s2_id = json.loads(msg.payload.decode('utf-8'))["sensorId"]

    if (s1_recieved and s2_recieved):
        s1_recieved = False
        s2_recieved = False
        worker = PredictionWorker(queue)
        worker.daemon = True
        worker.start()
        s1_temp = np.full((11,1,1),s1_temp)
        s2_temp = np.full((11,1,1),s2_temp)
        #s1_temp = np.array([[[s1_temp]]])
        s1 = np.concatenate([s1_array,s1_temp], axis=1)
        #s2_temp = np.array([[[s2_temp]]])
        s2 = np.concatenate([s2_array,s2_temp], axis=1)
        #s2 = np.full((11,1,1),s2_temp)
        s = np.concatenate([s1,s2], axis=2)
        #print("s1 ",s1.shape)
        #print("s2 ",s2.shape)
        #print("s ",s.shape)

        #testD_main_1 = np.concatenate([testD_main,s], axis=1)
        #print(testD_main_1.shape)
	    #data_input=new_input
	    #data_in1 = 3
	    #data_in2 = 4
	    #data_output=new_output
        queue.put((s, testO_main, s1_id,s2_id, sensor_time))


class PredictionWorker(Thread):
    
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
     
    def run(self):
        while True:
            testD_main, testO_main, s1_id,s2_id, sensor_time = self.queue.get()
            try:
                st = time()
                prediction, s1r_id,s2r_id, s_time, y_pred = inference(testD_main, testO_main,s1_id,s2_id,sensor_time)
                et = time()
                with open('times.txt','a') as f:
                    t = str((et-st)) + '\n'
                    f.write(t) 
                print("time taken: ",(et-st))
                dt = datetime.fromtimestamp(s_time)

                #print(prediction)
                ##if(np.any(prediction)):
                first_row = y_pred[1,:]
                #print(first_row)
                #print(first_row[0])
                #print(first_row[1])
                if(first_row[0] >= 0.7 and first_row[1] >= 0.7):
                    #print(dt, " : Normal Readings")
                    print(dt,f"{color.GREEN} Normal Readings: {color.ENDC}")
                    brokers_out = {"time":et,"sensor_1_id":s1r_id,"sensor_2_id":s2r_id}
                    data_out=json.dumps(brokers_out) # encode object to JSON
                    #print(data_out)
                   # client.publish("siloAnomaly/1",data_out)
                else:
                    print(dt,f"{color.RED} Anomaly detected!!! {color.ENDC}")
                    #print(dt, " : Anomaly detected!!")
                    brokers_out = {"time":et,"sensor_1_id":s1r_id,"sensor_2_id":s2r_id}
                    data_out=json.dumps(brokers_out) # encode object to JSON
                    print(data_out)
                    client.publish("siloAnomaly/1",data_out)
                    
                #print(type(prediction))

                
            finally:
                self.queue.task_done()
  
  



def main():

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("192.168.43.164", 1883, 60)
    client.loop_forever()

    queue.join()

if __name__ == '__main__':
    main()
