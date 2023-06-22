
import os
import cv2
import numpy as np
#from PIL import Image
#from pygame.locals import *
from threading import Thread

import mediapipe as mp

"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *
    *                          camStream setup                                  *
    *                                                                           *
    *___________________________________________________________________________*
"""
OPENCV_LOG_LEVEL=0
WIDTH, HEIGHT = 1080, 1920
GREEN = (0, 255, 0)
RED = (0, 0, 255)
X, Y = 0, 0
class WebcamStream:
    # initialization method
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 1 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        self.frame_ready = False
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())

        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            self.frame_ready = True

            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break

        self.vcap.release()




    # method to return latest read frame
    def read(self):
        return self.frame


    # method to stop reading frames
    def stop(self):
        self.stopped = True


hands = mp.solutions.hands.Hands()

body_landmarker = mp.solutions.pose.Pose()

drawing = mp.solutions.drawing_utils

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

# initializing and starting multi-threaded webcam input stream
webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0

# define a video capture object
# vid = cv2.VideoCapture(0)
p = np.zeros((21,3))
""""
    ___________________________________________________________________________
    *                                                                          *
    *                        ball class                                *
    *                                                                          *
    *__________________________________________________________________________*
"""
import imageio
from PIL import Image, ImageSequence


g = 1.8
class Ball:
    # [x: 0- HEIGHT, y: 0- WIDTH]
    # +--------------------+
    # | 0,0        0,WIDTH |
    # |                    |
    # |                    |
    # | HEIGHT,0       W,H |
    # +--------------------+
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

        self.v = np.array([0,-20],dtype=np.float64)
        self.score = 0
        self.high_score = HIGH_SCORE
        self.champ= CHAMP

    def update(self):
        self.x += self.v[0]
        self.y += self.v[1]
        self.v[1] += g
        self.check_collision()
        self.v *= 0.92
        #print(self.x,self.y)
    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.r), (0, 0, 255), -1)

    def check_collision(self):
        if self.x + self.r >= X or self.x - self.r <= 0:
            self.v[0] *= -1.5
            if self.x + self.r >= X:
                self.v += np.array([10, 0])
            else:
                self.v += np.array([10, 0])

        # floor bounce
        if self.y + self.r >= Y:
            self.v[1] *= -1
            self.v -= np.array([0, 10])
            self.y -= 10
            self.score = 0




        if self.y - self.r <= 0:
            self.v[1] *= -1
            self.v += np.array([0, 30])
            self.y += 10



    def check_collision_body(self, landmarks):
        for i in range(11, 33):
            if i in [23,24]:
                continue
            diff = np.array([landmarks[i,0], landmarks[i,1]]) - np.array([self.x, self.y])
            dist = np.linalg.norm(diff)
            if dist <= self.r and self.v[1] > -1:
                self.v -= 76*diff/dist

                self.score +=1
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.champ = name


                break

        return False



"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *
    *                               main Loop                                   *
    *                                                                           *
    *___________________________________________________________________________*
"""

file1 = open("high_score.txt","r")
HIGH_SCORE = int(file1.read())
file1.close()

file1 = open("champ.txt","r")
CHAMP = file1.read()
file1.close()

name = input("Enter your name: ")
p = np.zeros((33,3))
ball = Ball(500,100,60)
while (True):

    # Capture the video frame
    # by frame
    frame = cv2.flip(webcam_stream.read(),1)

    pose = body_landmarker.process(frame)

    #parse the landmarks
    if pose.pose_landmarks:
        drawing.draw_landmarks(frame, pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        for i, mark in enumerate(pose.pose_landmarks.landmark):
            p[i, 0], p[i, 1], p[i, 2] = mark.x * HEIGHT, mark.y * WIDTH, mark.z * HEIGHT

    cv2.putText(frame, f'Score: {ball.score}', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 250, 250), 6, cv2.LINE_AA)
    cv2.putText(frame, f'High Score: {ball.high_score} ({ball.champ})', (750, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 250, 250), 6, cv2.LINE_AA)

    ball.draw(frame)
    # Display the resulting frame
    cv2.imshow('output', frame)
    _, _, X, Y = cv2.getWindowImageRect('output')
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    ball.check_collision_body(p)
    ball. update()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if ball.high_score > HIGH_SCORE:
    file1 = open("high_score.txt", "w")
    file1.write(str(ball.high_score))
    file1.close()

    file1 = open("champ.txt", "w")
    file1.write(name[:10])
    file1.close()
    print("new high score")

# After the loop release the cap object
webcam_stream.vcap.release()
frame.release()
# Destroy all the windows
cv2.destroyAllWindows()