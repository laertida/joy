import os
#Removes logs from tensorflow to avoid unnecesary data in pipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pyaudio
#from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd
from functions import (checkThreshhold, logTaker, handleGPIO)
import datetime

import atexit

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from plot import Plotter

@atexit.register
def on_close():
	 handleGPIO(False)

if __name__ == "__main__":

    ################### SETTINGS ###################
    plt_classes = [0,13,14,15,16,17,18,494] # Speech, Laugh, Baby Laughter, Belly Laugh, Shockle Hurtle Silence 
    ejemplosRisa = [13,14,15,16,17,18]
    
    defaultRepeats = {
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0
    }
    
    repeats = defaultRepeats
    
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 500
    MIC = None

    # print(sd.query_devices())

    #################### MODEL #####################
    model = YAMNet(weights='keras_yamnet/yamnet.h5')
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

    #################### LOG VARIABLES  #####################
    timerFlag = False
    timer = 0
    timerMax = 3

    led = False

    startDate = ""
    endDate = ""

    threshhold = [0.6]

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    #monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Waveform detection
        data = preprocess_input(np.frombuffer(
            stream.read(CHUNK), dtype=np.float32), RATE)
        prediction = model.predict(np.expand_dims(data,0), verbose=0)[0]

        ########## LOGS MANAGEMENT

        #RESETS TIMER IF LAUGH IS DETECTED IN THE CYCLE
        if (timerFlag == True):
            if (checkThreshhold(threshhold[0], prediction[plt_classes[1]]) == True):
                predicted_class_index = next((idx for idx, val in enumerate(prediction) if val == max(prediction)), None)
                
                if predicted_class_index in ejemplosRisa:
                    repeats[predicted_class_index] += 1
                    timer = timerMax
            else:
                timer -= 1

        #END LAUGH LOG CYCLE, TURNS OF MOVEMENT AND REGISTERS LOG IN THE 
        if (timerFlag == True and timer <= 0):
            endDate = str(datetime.datetime.now())

            # log entry
            handleGPIO(False)

            logTaker(startDate, endDate, repeats)

            #RESET VALUES
            timerFlag = False
            startDate = ""
            endDate = ""
            
            repeats = defaultRepeats

        #START LAUGH CYCLE
        if (timerFlag == False and checkThreshhold(threshhold[0], prediction[plt_classes[1]]) == True):
            predicted_class_index = next((idx for idx, val in enumerate(prediction) if val == max(prediction)), None)
            
            if predicted_class_index in ejemplosRisa:  
                repeats[predicted_class_index] += 1
                # set values
                startDate = str(datetime.datetime.now())
                timerFlag = True
                timer = timerMax
                
                # on signal
                handleGPIO(True)

        #monitor(data.transpose(), np.expand_dims(prediction[plt_classes],-1))

    # close audio streams 
    stream.stop_stream()
    stream.close()
    audio.terminate()
