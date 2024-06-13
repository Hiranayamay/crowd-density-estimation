import cv2
import numpy as np
import pandas as pd
import argparse

# Argument parser for dynamic video source input
parser = argparse.ArgumentParser(description='Crowd Density Estimation')
parser.add_argument('--video', type=str, default='crowd2.mp4', help='Path to the video file or camera index (0 for webcam)')
args = parser.parse_args()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
try:
    video_source = int(args.video)
except ValueError:
    video_source = args.video

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: Could not open video source {args.video}.")
    exit()

# Variables to store counts and timestamps
people = []
time_counter = []
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image or end of video")
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and count people
    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            people_count += 1

    # Logging count and timestamp
    people.append(people_count)
    time_counter.append(counter)
    counter += 1

    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# Save results to DataFrame and print
table = {'people': people, 'time': time_counter}
df = pd.DataFrame(table, columns=['people', 'time'])
print(df)

# Optional: Export DataFrame to CSV
# df.to_csv('export_dataframe1.csv', index=False, header=True)

# Optional: Display the data in a plot (requires matplotlib)
# df.plot(x='time', y='people', kind='line')
# plt.show()
