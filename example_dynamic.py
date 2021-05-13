from cv2 import cv2
import numpy as np
from PQFTLib import PQFT
import time
import pandas as pd

video_name = "rolling_resized.mp4"
video_path = "./example/"


def read_video_stream(name):
    frames = list()
    cap = cv2.VideoCapture(name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
    cap.release()
    return frames


frames = read_video_stream(video_path+video_name)

prev_frame = frames[0]
for i in range(len(frames)-1):
    this_frame = np.array(frames[i])
    cv2.imshow(f"frame {str(i)}", this_frame)
    cv2.waitKey(0)
    saliency_map = PQFT(prev_frame, this_frame)
    cv2.imshow(f"saliency map{str(i)}", saliency_map)
    cv2.waitKey(0)
    prev_frame = np.array(frames[i])

print("END")
