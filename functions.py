#Function used to create a log when a value over the threshold is detected
def checkThreshhold(threshold, value):
    if (value > threshold):
            #print("ruido!")
            return True
    else:
          return False
    
def logTaker(start, end, count, thresholds):
      log = str(start) + ", " + str(end) + ", " + str(count) 

      for i in thresholds:
            log = log + ", " + str(i)

      print(log)      