import os
from queue import Queue
from threading import Thread
from time import time
import pandas as pd
import paho.mqtt.client as mqtt
from main2 import inference
from test import test_function
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

#model_name = "clean_tfrt.pb"
#queue = Queue()

pool = ThreadPool(2) ##8


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("SCRI/jej")

def on_message(client, userdata, msg):
    print("got something!")
    #data_input = np.array([[0.068428279,0,0.889573295,0.363636364,0.746389433,0.010989011]])
    testD_main = np.full((1000,10,2), 10)
    #testO_main = np.full((1000,2), 10)
    #data_input=str(msg.payload)
    #queue.put((model_name, data_input))
    #pool.map(predictor,testD_main)
    pool.map(predictor,testD_main)
    #pool.close()
   # pool.join()


def predictor(testD):
    testO = np.full((1000,2), 10)
    #results = inference(testD,testO)
    #prediction = inference(testD,testO)
    x = test_function()
    print(x)
    #print(results)

# class PredictionWorker(Thread):

#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue

#     def run(self):
#         while True:
#             model_name, data_input = self.queue.get()
#             try:
#                 prediction = predict_power(model_name, data_input)
#                 print("The label added is: ", prediction)
#             finally:
#                 self.queue.task_done()
                
def main():

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("broker.mqttdashboard.com", 1883, 60)
    client.loop_forever()


if __name__ == '__main__':
    main()
