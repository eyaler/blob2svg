import numpy as np
import cv2
from scipy import ndimage
from itertools import groupby

def blob2svg(image, blob_levels=(1,255), method=None, label=None, color=None, save_to=None, show=False):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    image = image.squeeze()
    assert len(image.shape) == 2, 'image must be single channel'

    try:
        len(blob_levels)
    except:
        blob_levels = [blob_levels]
    if len(blob_levels) == 1:
        blob_levels = [blob_levels[0]]*2
    assert len(blob_levels) == 2
    image = ((image >= min(blob_levels)) * (image <= max(blob_levels)) * 255).astype(np.uint8)
    zoom = ndimage.zoom(image, 2, order=0) # this is in order to get a scalable vertex-following contour instead of a pixel-following contour

    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE

    _, all_contours, _ = cv2.findContours(zoom, mode=cv2.RETR_LIST, method=method)
    _, contours, _ = cv2.findContours(zoom, mode=cv2.RETR_EXTERNAL, method=method)
    if len(contours)!=len(all_contours):
        print('Warning: found and ignored %d child contours'%(len(all_contours)-len(contours)))

    svg = []
    if color is None:
        r = lambda: np.random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
    if label is None:
        label = color

    contours.sort(key=lambda x: str(x))
    for i,c in enumerate(contours):
        c = [((p[0][0] + 1) // 2, (p[0][1] + 1) // 2) for p in c]

        # remove consecutive duplicates
        c = [x[0] for x in groupby(c)]
        if len(c)>1 and c[-1]==c[0]:
            del c[-1]

        points = ' '.join(str('%d,%d'%(p[0], p[1])) for p in c)
        svg.append('<polygon class="%s" fill="%s" id="%d" points="%s"/>'%(label, color, i, points))

    if save_to is not None:
        save_svg(save_to, svg, resolution=image.shape[::-1])

    if show:
        cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cimage, [c//2 for c in contours], -1, color=(0, 0, 255))
        cv2.imshow('Found Contours', cimage)
        cv2.waitKey()

    return svg

def save_svg(filename, svg, resolution=None):
    if resolution is None:
        resolution = ''
        print('Warning: resolution not specified')
    else:
        resolution = 'width="%s" height="%s"'%(resolution[0], resolution[1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" version="1.1" %s>'%(resolution)] + svg + ['</svg>']
    with open(filename, 'w') as f:
        f.writelines(s+'\n' for s in svg)
