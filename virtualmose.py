import mediapipe as mp
import numpy as np
import cv2
import random
from math import sqrt
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
click = 0

video = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        imageHeight, imageWidth, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        indexfingertip_x = indexfingertip_y = None
        thumbfingertip_x = thumbfingertip_y = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(250, 44, 0), thickness=2, circle_radius=2),
                )

                for point in mp_hands.HandLandmark:
                    normalized_landmark = hand_landmarks.landmark[point]
                    pixel_coordinates = mp_drawing._normalized_to_pixel_coordinates(
                        normalized_landmark.x, normalized_landmark.y, imageWidth, imageHeight
                    )

                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        if pixel_coordinates is not None:
                            indexfingertip_x, indexfingertip_y = pixel_coordinates
                            pyautogui.moveTo(indexfingertip_x*3, indexfingertip_y*4)
                    elif point == mp_hands.HandLandmark.THUMB_TIP:
                        if pixel_coordinates is not None:
                            thumbfingertip_x, thumbfingertip_y = pixel_coordinates

        if indexfingertip_x is not None and thumbfingertip_x is not None:
            distance = sqrt((indexfingertip_x - thumbfingertip_x) ** 2 + (indexfingertip_y - thumbfingertip_y) ** 2)
            if distance < 20:
                click += 1
                if click % 5 == 0:
                    print("Single click")
                    pyautogui.click()

        cv2.imshow("HandMouse", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
