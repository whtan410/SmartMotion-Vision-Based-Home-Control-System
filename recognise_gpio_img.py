#Import library
import cv2
import imutils
import numpy as np
import tensorflow
import RPi.GPIO as GPIO
from rpi_lcd import LCD
import time

#Initialise LCD
display = LCD()
##set GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#set GPIO as output
GPIO.setup(19,GPIO.OUT) #LED1
GPIO.setup(26,GPIO.OUT) #LED2
GPIO.setup(21,GPIO.OUT) #BUZZER
GPIO.setup(5,GPIO.OUT)  #MOTOR+END
GPIO.setup(6,GPIO.OUT)  #MOTOR-END

#Load tensorflow lite model
model = tensorflow.lite.Interpreter(model_path=r"/home/pi/Desktop/V11/gestpred_opt.tflite")

#Allocate tensor
model.allocate_tensors()

#Get input and output tensors
in_dets = model.get_input_details()
out_dets = model.get_output_details()

bkgd = None
def sub_bkgd(img, weightaccum):
    global bkgd
    if bkgd is None:
        bkgd = img.copy().astype("float")
        return
    cv2.accumulateWeighted(img, bkgd, weightaccum)

def segmentize(img, threshold=35):
    global bkgd
    bs = cv2.absdiff(bkgd.astype("uint8"), img)
    thresholded = cv2.threshold(bs, threshold, 255, cv2.THRESH_BINARY)[1]
    (contours, hierarchy) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        hand_segmented = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segmented)

def gestpred(model):
    #Preprocessing image
    img = cv2.imread("Temp.png")
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayimg, (64, 64))
    reshaped = resized.reshape(1, 64, 64, 1)

    ## Predicting the gesture number
    input_shape = in_dets[0]['shape']
    in_ten = np.array(reshaped, dtype=np.float32)
    in_idx = model.get_input_details()[0]["index"]
    model.set_tensor(in_idx, in_ten)
    model.invoke()
    out_dets = model.get_output_details()
    output_data = model.get_tensor(out_dets[0]['index'])
    prediction = np.squeeze(output_data)
    gestnum = np.argmax(prediction)

    ## Function corresponds to gesture number
    if gestnum == 0:
        status = 0
    elif gestnum == 1:
        status = 1
    elif gestnum == 2:
        status = 2
    elif gestnum == 3:
        status = 3
    elif gestnum == 4:
        status = 4
    elif gestnum == 5:
        status = 5
    elif gestnum == 6:
        status = 6
    elif gestnum == 7:
        status = 7

    ##Perform action according to status
    if status == 0:
        GPIO.output(19, GPIO.LOW)
        GPIO.output(26, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.LOW)
        return "FIST"

    elif status == 1:
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(26, GPIO.HIGH)
        GPIO.output(21, GPIO.LOW)
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.LOW)
        return "ONE"

    elif status == 2:
        GPIO.output(19, GPIO.LOW)
        GPIO.output(26, GPIO.LOW)
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.LOW)
        return "TWO"

    elif status == 3:
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(26, GPIO.HIGH)
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(5, GPIO.LOW)
        GPIO.output(6, GPIO.LOW)
        return "THREE"

    elif status == 4:
        GPIO.output(19, GPIO.LOW)
        GPIO.output(26, GPIO.LOW)
        GPIO.output(21, GPIO.LOW)
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        return "THUMBSUP"

    elif status == 5:
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(26, GPIO.HIGH)
        GPIO.output(21, GPIO.LOW)
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        return "FIVE"

    elif status == 6:
        GPIO.output(19, GPIO.LOW)
        GPIO.output(26, GPIO.LOW)
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        return "SIX"

    elif status == 7:
        GPIO.output(19, GPIO.HIGH)
        GPIO.output(26, GPIO.HIGH)
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(5, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        return "SEVEN"

#Main function
if __name__ == "__main__":
    #Initialise camera and frame counting
    weightaccum = 0.5
    cam = cv2.VideoCapture(0)
    framecount = 0
    count = 0

    #Main function
    while(True):
        ret, frame = cam.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        copy = frame.copy()
        roi = frame[10:250, 300:550]
        cvtgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cvtgray, (7, 7), 0)

        if framecount < 35:
            sub_bkgd(blur, weightaccum)
            if framecount == 1:
                print("Calibrating")
            elif framecount == 34:
                print("Calibrated")

        else:
            palm = segmentize(cvtgray)

            if palm is not None:
                (thresholded, hand_segmented) = palm
                cv2.drawContours(copy, [hand_segmented + (300, 10)], -1, (0,0,255))

                if cv2.waitKey(1) == ord('q'):
                        cv2.imwrite('Temp.png',thresholded)
                        cv2.imshow('Temp.png',thresholded)
                    
        gestname = gestpred(model)
        cv2.putText(copy, str(gestname), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 2)
        display.text("Gesture:",1)
        display.text(gestname,2)

        count = count + 1
        cv2.rectangle(copy, (550, 10), (300, 250), (0,0,255), 1)
        framecount += 1
        cv2.imshow("Webcam", copy)
        interrupt_key = cv2.waitKey(1) & 0xFF
        if interrupt_key == 27:
            break

##Clear all active window
display.clear()
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()