import 	subprocess

#Function used to create a log when a value over the threshold is detected
def checkThreshhold(threshold, value):
    if (value > threshold):
            return True
    else:
          return False

def handleGPIO(status):
    # TODO: implement in native Python GPIO
    if (status == True):
        #print("on")
        subprocess.run("./gpio/ledOn.sh", shell=True)
    else:
        #print("off")
        subprocess.run("./gpio/ledOff.sh", shell=True)


def logTaker(start, end, repeatsDict):
    elements = [start, end] + list(repeatsDict.values())

    log = ",".join(str(i) for i in elements) 

      #log = str(start) + ", " + str(end) + ", " + str(count) 
      #for i in thresholds:
      #      log = log + ", " + str(i)

    print(log)      
