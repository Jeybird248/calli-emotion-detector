import cv2
import os
import numpy as np


def detect_face(img):
    faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('face-detector.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10)
    if len(faces) == 0:
        return None, None
    for (x, y, w, h) in faces:
        return gray[y:y + w, x:x + h], faces[0]


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


img1 = cv2.imread("test-data/2.png")

face, rect = detect_face(img1)
draw_rectangle(img1, rect)
cv2.imshow("Test Image 1", img1)
cv2.waitKey(0)
