import torch
import numpy as np
import cv2
import time
import telebot
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from datetime import datetime

class IntrusionDetection:
    """
    Class implements YoloV5 model to detect Intrusion from Camera
    """

    def __init__(self, url, token, receiver_id, url_of_group, chat_id, out_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.token = token
        self.receiver_id = receiver_id
        self.bot = telebot.TeleBot(token)
        self.out_file = out_file
        self.url_of_group = url_of_group
        self.chat_id = chat_id
        self.image_coordinates = []
        self.right_click_happened = False
        self.count = 0
        self.var = True
        print("\n\nDevice Used:", self.device)

    def get_video_from_url(self):
        return cv2.VideoCapture(self._URL)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6 and self.class_to_label(labels[i]) == 'person':
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                text = self.class_to_label(labels[i])
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def sending_to_telegram(self, results, frame):
        labels, cord = results
        n = len(labels)

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6 and self.class_to_label(labels[i]) == 'person':
                self.count += 1
                if self.count == 20:
                    self.count = 0
                    self.var = True

                if self.var:
                    self.var = False
                    # Get current timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    message = f"🚨 Person detected at {timestamp}"

                    # Send text message
                    text_url = f"{self.url_of_group}/sendMessage?chat_id={self.chat_id}&text={message}"
                    requests.get(text_url)

                    # Save the frame as an image
                    image_path = "alert.jpg"
                    cv2.imwrite(image_path, frame)

                    # Send the image
                    with open(image_path, 'rb') as photo:
                        files = {'photo': photo}
                        image_url = f"{self.url_of_group}/sendPhoto"
                        data = {'chat_id': self.chat_id}
                        requests.post(image_url, files=files, data=data)



    def to_send_or_not(self, results):
        labels, cord = results
        n = len(labels)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6 and self.class_to_label(labels[i]) == 'person':
                return True
        return False

    def extract_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.image_coordinates) < 2:
                self.image_coordinates.append([x, y])


    def call(self):
        player = self.get_video_from_url()
        assert player.isOpened()
    
        # Notify that the camera is active
        base_url = self.url_of_group + f'/sendMessage?chat_id={self.chat_id}&text=Your Camera is Active Now.'
        requests.get(base_url)

        while True:
            ret, frame = player.read()
            if not ret:
                break

            param = frame.copy()
            cv2.imshow('image', frame)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

            # ROI selection using 2 mouse clicks
            if len(self.image_coordinates) < 2:
                cv2.setMouseCallback('image', self.extract_coordinates, param)
                for idx, point in enumerate(self.image_coordinates):
                    x, y = point
                    cv2.circle(frame, center=(x, y), radius=6, color=(0, 0, 255), thickness=-1)
                    cv2.putText(frame, f"P{idx+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)


                if len(self.image_coordinates) == 1:
                    cv2.putText(frame, "Click bottom-right corner", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif len(self.image_coordinates) == 0:
                    cv2.putText(frame, "Click top-left corner", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow('image', frame)
                continue

            # Define ROI rectangle
            pt1 = tuple(self.image_coordinates[0])
            pt2 = tuple(self.image_coordinates[1])
            x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
            x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])

            # Draw rectangle on full frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            roi = frame[y1:y2, x1:x2]
            cv2.imshow('image', frame)

            # Run YOLO detection on ROI
            results = self.score_frame(roi)

            # Show detection boxes in ROI
            roi_with_boxes = self.plot_boxes(results, roi.copy())
            cv2.imshow("roi", roi_with_boxes)

            # Optionally show detections on full frame
            frame_with_boxes = self.plot_boxes(results, frame.copy())
            cv2.imshow("image", frame_with_boxes)

            # If person detected, send alert
            if self.to_send_or_not(results):
                self.sending_to_telegram(results, frame)


        player.release()
        cv2.destroyAllWindows()



# IP Webcam: 'http://192.168.43.1:8080/video'
# Creating a new object and executing
url_of_camera = 0  # Use 0 for default webcam
detection = IntrusionDetection(
    url_of_camera,
    '8104550765:AAGnkh0m_A_JzHKjdvBhcJLnR_db1hfnSYQ',
    '8111720515',
    'https://api.telegram.org/bot8104550765:AAGnkh0m_A_JzHKjdvBhcJLnR_db1hfnSYQ',
    '-4826787436',
    "video2.avi"
)

detection.call()
cv2.destroyAllWindows()
