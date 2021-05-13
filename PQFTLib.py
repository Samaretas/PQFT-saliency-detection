from cv2 import cv2
import numpy as np


def PQFT(prev_img, next_img, map_size = 64):
    """
        Computing saliency using phase spectrum of quaternion fourier transform.
        Images are 3 channel.
    """

    new_shape = (int(next_img.shape[1]/(next_img.shape[0]/map_size)), map_size)
    next_img = cv2.resize(next_img, new_shape, cv2.INTER_LINEAR)
    (b, g, r) = cv2.split(next_img)

    # color channels
    R = r-(g+b)/2
    G = g-(r+b)/2
    B = b-(r+g)/2
    Y = (r+g)/2-abs(r-g)/2-b

    red_green = R-G
    blue_yellow = B-Y

    # intensity
    intensity = np.sum(next_img, axis=-1)

    # motion
    prev_img = cv2.resize(prev_img, new_shape, cv2.INTER_LINEAR)
    prev_intensity = np.sum(prev_img, axis=-1)
    movement = abs(intensity-prev_intensity)

    planes = [
        movement.astype(np.float64),
        red_green.astype(np.float64),
    ]
    f1 = cv2.merge(planes)
    f1 = cv2.dft(f1)
    planes = cv2.split(f1)

    magnitude1 = cv2.magnitude(planes[0], planes[1])
    magnitude1 = cv2.multiply(magnitude1, magnitude1)

    planes = [
        blue_yellow.astype(np.float64),
        intensity.astype(np.float64),
    ]

    f2 = cv2.merge(planes)
    f2 = cv2.dft(f2)
    planes = cv2.split(f2)

    magnitude2 = cv2.magnitude(planes[0], planes[1])
    magnitude2 = cv2.multiply(magnitude2, magnitude2)

    magnitude = magnitude1+magnitude2
    magnitude = cv2.sqrt(magnitude)

    planes[0] = planes[0]/magnitude
    planes[1] = planes[1]/magnitude
    f2 = cv2.merge(planes)

    planes = cv2.split(f1)
    planes[0] = planes[0]/magnitude
    planes[1] = planes[1]/magnitude
    f1 = cv2.merge(planes)

    cv2.dft(f1, f1, cv2.DFT_INVERSE)
    cv2.dft(f2, f2, cv2.DFT_INVERSE)

    planes = cv2.split(f1)
    magnitude1 = cv2.magnitude(planes[0], planes[1])
    magnitude1 = cv2.multiply(magnitude1, magnitude1)

    planes = cv2.split(f2)
    magnitude2 = cv2.magnitude(planes[0], planes[1])
    magnitude2 = cv2.multiply(magnitude2, magnitude2)

    magnitude = magnitude1 + magnitude2

    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 8, None, 8)
    saliency = np.zeros((new_shape[0], new_shape[1], 1), np.uint8)
    saliency = cv2.normalize(magnitude, saliency, 0,
                             255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return saliency
