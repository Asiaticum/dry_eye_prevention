import time
import os
import sys
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance


def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return eye_ear


cap = cv2.VideoCapture("media/blink.mp4")
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # グレースケールの方が高速に検出できる

# 顔部分の切り出し
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
x, y, w, h = faces[0, :]
face_gray = gray_frame[y:(y + h), x:(x + w)]
scale = 480 / h
face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

# dlib.rectangle(long(left), long(top), long(right), long(bottom))
face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
face_parts = face_parts_detector(face_gray_resized, face)
face_parts = face_utils.shape_to_np(face_parts)

right_eye = face_parts[36:42]
left_eye = face_parts[42:48]
ear = (calc_ear(right_eye) + calc_ear(left_eye)) / 2

print(faces)
