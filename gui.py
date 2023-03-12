import cv2
import numpy as np
from tkinter import *

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
# Define the classes
classes = []
with open("class.names", "r") as f:
    classes = f.read().splitlines()

def start_detection():
    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        height, width, _ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
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
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Motion Detection", frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

def stop_detection():
    global running
    running = False



# Create GUI
root = Tk()
root.title("Motion Detection")
root.geometry("200x50")

start_button = Button(root, text="Start", command=start_detection)
start_button.pack()

stop_button = Button(root, text="Stop", command=stop_detection)
stop_button.pack()

root.mainloop()