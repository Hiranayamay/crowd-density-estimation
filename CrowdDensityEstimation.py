import cv2
import numpy as np
import pandas as pd

people = []
time_counter = []
counter = 0
table = {'people': [], 'time': []}

kernel1 = np.ones((2, 2), np.float32) / 4

# Change video source here
VIDEO_SRC = 'crowd2.mp4'
cap = cv2.VideoCapture(VIDEO_SRC)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_SRC}.")
    exit()

# Read the first frame as the background
ret, back = cap.read()
if not ret:
    print("Error: Could not read the video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image or end of video")
        break

    imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(imggray, 9)
    ret, imgthresh = cv2.threshold(med, 100, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.dilate(opening, kernel1, iterations=1)

    count = (cv2.countNonZero(closing)) / 1700

    if counter % 5 == 0:
        people.append(int(count))
        time_counter.append(counter)
    counter += 1
    if counter == 100:
        break

    h, w = frame.shape[:2]
    contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# Save results to DataFrame and print
table['people'] = people
table['time'] = time_counter

df = pd.DataFrame(table, columns=['people', 'time'])
print(df)

# Optional: Export DataFrame to CSV
# df.to_csv(r'export_dataframe1.csv', index=False, header=True)

# Optional: Display the data in a plot (requires matplotlib)
# df.plot(x='time', y='people', kind='line')
# plt.show()

def checkDistance():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image or end of video")
        return

    imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(imggray, 9)
    ret, imgthresh = cv2.threshold(med, 100, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.dilate(opening, kernel1, iterations=1)
    # Add further processing here as needed
