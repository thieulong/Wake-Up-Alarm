import cv2
import mediapipe as mp
import os
import time
import datetime
import sys

try:

  hour = int(sys.argv[1])
  minute = int(sys.argv[2])

except IndexError:

  print("[ERROR] Please run the script in the terminal as: python alarm.py hh mm")
  print("[NOTE] Value hh and mm will be replaced by your desired alarm time (hours and minutes)")
  print("[NOTE] Please enter 0 in the front if value is 1 digit")

else:

  from pygame import mixer

  mixer.init()

  os.system('clear')

  if hour < 12:
        
        print("[INFO] Alarm set up successfully at {}: {} AM".format(hour, minute))

  elif hour >= 12:
        
        print("[INFO] Alarm set up successfully at {}: {} PM".format(hour, minute))

  mixer.music.load('ring.wav')

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose

  cap = cv2.VideoCapture(0)

  while True:

    if datetime.datetime.now().hour == hour and datetime.datetime.now().minute == minute:

      with mp_pose.Pose(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
          success, image = cap.read()
          image_height, image_width, _ = image.shape

          mixer.music.play()

          if not success:
            print("Ignoring empty camera frame.")
            continue

          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = pose.process(image)

          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          # mp_drawing.draw_landmarks(
          #     image,
          #     results.pose_landmarks,
          #     mp_pose.POSE_CONNECTIONS,
          #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

          if results.pose_landmarks:
                
              x_cordinate = list()
              y_cordinate = list()

              for id, lm in enumerate(results.pose_landmarks.landmark):
                  h, w, c = image.shape
                  cx, cy = int(lm.x * w), int(lm.y * h)
                  x_cordinate.append(cx)
                  y_cordinate.append(cy)
                  
              cv2.rectangle(img= image,
                            pt1= (min(x_cordinate), max(y_cordinate)),
                            pt2 = (max(x_cordinate), min(y_cordinate)-20),
                            color= (0,0,255),
                            thickness= 2)

              # Detect standing with hands over head
              left_side = [results.pose_landmarks.landmark[28].y * image_height,
                          results.pose_landmarks.landmark[26].y * image_height,
                          results.pose_landmarks.landmark[24].y * image_height,
                          results.pose_landmarks.landmark[12].y * image_height,
                          results.pose_landmarks.landmark[0].y * image_height,
                          results.pose_landmarks.landmark[16].y * image_height]

              right_side = [results.pose_landmarks.landmark[27].y * image_height,
                            results.pose_landmarks.landmark[25].y * image_height,
                            results.pose_landmarks.landmark[23].y * image_height,
                            results.pose_landmarks.landmark[11].y * image_height,
                            results.pose_landmarks.landmark[0].y * image_height,
                            results.pose_landmarks.landmark[15].y * image_height]

              left_straight = all(i > j for i, j in zip(left_side, left_side[1:]))
              right_straight = all(i > j for i, j in zip(right_side, right_side[1:]))

              if left_straight and right_straight:
                    
                cv2.rectangle(img= image,
                pt1= (min(x_cordinate), max(y_cordinate)),
                pt2 = (max(x_cordinate), min(y_cordinate)-20),
                color= (0,255,0),
                thickness= 2)

                mixer.music.stop() 

                mixer.music.load('confirm.wav')

                mixer.music.play()

                time.sleep(1)

                break

              x_cordinate.clear()
              y_cordinate.clear()

          cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
          if cv2.waitKey(5) & 0xFF == 27:
            break
      cap.release()