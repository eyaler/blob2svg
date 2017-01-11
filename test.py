import cv2
from blob2svg import blob2svg

img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
svg = blob2svg(img, filename='test.svg')
