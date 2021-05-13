from cv2 import cv2
import matplotlib.pyplot as plt
from PQFTLib import PQFT

image = cv2.imread('example/my_mug.jpg')
shape = image.shape

saliency_map = PQFT(image, image, shape[0])

plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(saliency_map, 'gray')
plt.title('Saliency map')

plt.show()