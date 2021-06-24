# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tkinter

import autopy as autopy
import cv2
import mediapipe as mp
import winsound

import mouse as mouse

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# For static images:
IMAGE_FILES = ['C:/Users/tmnd/Desktop/leapGestRecog/00/01_palm/frame_00_01_0001.png']
originalPath = 'C:/Users/tmnd/Desktop/leapGestRecog/00/01_palm/frame_00_01_0004.png'
output_path = 'C:/Users/tmnd/Desktop/Saved landmark images/Example1.png'
output_path2_withText = 'C:/Users/tmnd/Desktop/Saved landmark images/Example2.png'

hand_open = 1
frequency = 1500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
#winsound.Beep(frequency, duration)

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        with open('landmark.txt','w') as landmark_txt:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_txt.write(str(hand_landmarks))
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


#-----------Save Image with Text--------------
            image = cv2.flip(annotated_image, 1)
            #image = cv2.imread(originalPath)
            # Window name in which image is displayed
            window_name = 'Image'

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (50, 50)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (0, 255, 0)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            image = cv2.putText(image, 'Hello', org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(
            #  '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            output_path2_withText, image)
#-----------Save Image with Text --------------

#-----------Save Image without Text --------------
        cv2.imwrite(
            #  '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
            output_path, cv2.flip(annotated_image, 1))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)
        print('Handendness',results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:

# --------------------------- Mouse Double Click when palm closed --------------------------
            if(hand_landmarks.landmark[0].y - hand_landmarks.landmark[12].y) > 0.2:
                if hand_open < 2:
                    hand_open = 2


            if (hand_landmarks.landmark[0].y - hand_landmarks.landmark[12].y) < 0.2:
                if hand_open > 1:
                    mouse.double_click('left')
                    #winsound.Beep(frequency, duration)
                    hand_open = 1

# --------------------------- Mouse Double Click when palm closed --------------------------

            print('hand_landmarks:', hand_landmarks)
            print(
            f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width},'
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
