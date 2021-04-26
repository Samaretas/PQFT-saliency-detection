from cv2 import cv2
import numpy as np
from PQFTLib import PQTF

video_name = "example\\rolling_resized.mp4"

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

frames = read_video_stream(video_name)

savepath = "example\\saliency_maps\\"
for i in range(len(frames)-1):
    # cv2.imshow(f"frame {str(i)}", frames[i])
    # cv2.waitKey(0)
    prev_frame = np.array(frames[i])
    this_frame = np.array(frames[i+1])

    saliency_map = PQTF(prev_frame, this_frame, 256)
    cv2.imwrite(savepath+"frame"+str(i)+"-"+str(i+1)+".jpg", saliency_map)
    # cv2.imshow(f"saliency map{str(i)}-{str(i+1)}", saliency_map)
    # cv2.waitKey(0)
    