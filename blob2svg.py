import numpy as np
import cv2

def blob2svg(image, blob_levels=(1,255), label=None, color=None, filename=None):
    image = image.squeeze()
    assert len(image.shape) == 2, 'Image must be single channel'

    try:
        len(blob_levels)
    except:
        blob_levels = [blob_levels]
    if len(blob_levels) == 1:
        blob_levels = [blob_levels[0]]*2
    assert len(blob_levels) == 2
    image = ((image >= min(blob_levels)) * (image <= max(blob_levels))).astype(np.uint8)

    _, all_contours, _ = cv2.findContours(image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=len(all_contours):
        print('Warning: found and ignored %d child contours'%(len(all_contours)-len(contours)))

    svg = []
    if color is None:
        r = lambda: np.random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
    if label is None:
        label = color

    svg.append('<class>%s</class>'%(label))
    for i,c in enumerate(contours):
        points = ' '.join(str('%d,%d'%(p[0][0], p[0][1])) for p in c)
        svg.append('<polygon class="%s" fill="%s" id="%d" points="%s" />'%(label, color, i, points))

    if filename is not None:
        save_svg(filename, svg)

    return svg

def save_svg(filename, svg):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'] + svg + ['</svg>']
    with open(filename, 'w') as f:
        f.writelines(s+'\n' for s in svg)
