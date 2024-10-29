import os
#Removes logs from tensorflow to avoid unnecesary data in pipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd
from functions import (checkThreshhold, logTaker)
import datetime


from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from plot import Plotter

if __name__ == "__main__":

    ################### SETTINGS ###################
    #plt_classes = [0,13,14,17] # Speech, Music, Explosion, Silence 
    plt_classes = [13,494]
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 500

    #print(sd.query_devices())
    MIC = None

    #################### MODEL #####################
    
    model = YAMNet(weights='keras_yamnet/yamnet.h5')
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')


    #################### LOG VARIABLES  #####################

    timerFlag = False
    timer = 0
    timerMax = 3

    laughCounter = 0

    led = False

    startDate = ""
    endDate = ""

    threshhold = [0.6]

    print("Start Date | End Date | Count | Threshold1 | Threshold2 | ...")

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    #print(str(datetime.datetime.now()))

    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Waveform
        data = preprocess_input(np.frombuffer(
            stream.read(CHUNK), dtype=np.float32), RATE)
        #prediction = model.predict(np.expand_dims(data,0))[0] version con barra de progreso
        prediction = model.predict(np.expand_dims(data,0), verbose=0)[0]

        #print("laughter",prediction[plt_classes[0]])
        #print("silence", prediction[plt_classes[1]])

        ########## LOGS MANAGEMENT

        #RESETS TIMER IF LAUGH IS DETECTED IN THE CYCLE
        if (timerFlag == True):
            if (checkThreshhold(threshhold[0], prediction[plt_classes[1]]) == True):
                timer = timerMax
                laughCounter += 1
            else:
                timer -= 1

        #END LAUGH LOG CYCLE, TURNS OF MOVEMENT AND REGISTERS LOG IN THE 
        if (timerFlag == True and timer <= 0):
            timerFlag = False

            endDate = str(datetime.datetime.now())
            # Cambiar por codigo de GPIO
            led = False
            #Mandar logs
            logTaker(startDate, endDate, laughCounter, threshhold)

            #RESET VALORES
            timerFlag = False
            laughCounter = 0
            startDate = ""
            endDate = ""

        #START LAUGH CYCLE
        if (timerFlag == False and checkThreshhold(threshhold[0], prediction[plt_classes[1]]) == True):
            timerFlag = True

            startDate = str(datetime.datetime.now())

            laughCounter += 1

            timer = timerMax

            # Cambiar por codigo de GPIO
            led = True

        #print("timer: " + str(timer))

        ##############



        monitor(data.transpose(), np.expand_dims(prediction[plt_classes],-1))

    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
