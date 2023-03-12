
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import pygame
import time

#pygame.mixer.init()

# Load the sound
#sound = pygame.mixer.Sound("alarm.WAV")

# Function to play the sound
#def play_sound():
#   sound.play()


net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open("class.names", "r") as f:
    classes = f.read().splitlines()


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

# Function for importing video
def import_video():
    filename = filedialog.askopenfilename(title="Select video file", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))
    cap = cv2.VideoCapture(filename)

    while True:
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                confidence = str(round(confidences[i], 2))

                if i < len(colors):
                    color = colors[i]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)



                cv2.imshow('Image', img)
        cap.release()






fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

def start_detection():

    # Start video capture
    cap = cv2.VideoCapture(0)
    while True:





        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        # cofidence rating and box to show what was detected starts
        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)


                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)


        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                #if label == 'person':
                    # Write the current frame to the output video


                #print("Person detected")
                    #play_sound()
                    #out.write(img)

                confidence = str(round(confidences[i],2))

                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                # cofidence rating and box to show what was detected starts


        cv2.imshow('Image', img)
        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()

    # to stop the detection
def stop_detection():
 cv2.destroyAllWindows()



# Create GUI
root = Tk()
root.title("Motion Detection")
root.geometry("300x100")

start_button = Button(root, text="Start", command=start_detection)
start_button.pack()

stop_button = Button(root, text="Stop", command=stop_detection)
stop_button.pack()

import_button = Button(root, text="Import Video", command=import_video)
import_button.pack()


root.mainloop()