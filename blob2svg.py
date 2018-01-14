import numpy as np
import cv2
from itertools import groupby
from time import time

def get_area(contour, i):
    p1 = contour[i-1][0]
    p2 = contour[i][0]
    p3 = contour[i+1 if i+1<len(contour) else 0][0]
    return abs((p2[0]-p1[0])*(p3[1]-p1[1])-(p3[0]-p1[0])*(p2[1]-p1[1]))/2

def vw_closed(contour, area):
    contour = np.asarray([x[0] for x in groupby(contour.tolist())]) # remove consecutive duplicates - for performance
    areas = [get_area(contour, i) for i in range(len(contour))]
    while len(contour)>2:
        min_area = min(areas)
        if min_area > area:
            break
        i = np.argmin(areas)
        contour = np.delete(contour, i, axis=0)
        del areas[i]
        if len(areas)>0:
            areas[i-1] = get_area(contour, i-1)
            j = i if i<len(areas) else 0
            areas[j] = get_area(contour, j)
    return contour

def blob2svg(image, blob_levels=(1, 255), approx_method=None, simp_method='VW', abs_eps=0, rel_eps=0, min_area=0, box=False, erode_dilate_iters=0, erode_dilate_kernel=np.ones((3,3)), erode_dilate_connectivity=8, label=None, color=None,
             save_to=None, save_png=False, png_bg_color=None, show=False, verbose=True):
    # approx_method can be one of: cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE (or None, default), cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS
    # simp_method can be one of: 'RDP', 'VW'
    # connectivity can be one of: 0, 4, 8

    start_time = time()
    np.random.seed(0)

    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    image = image.squeeze()
    assert len(image.shape) == 2, 'image must be single channel'

    try:
        len(blob_levels)
    except:
        blob_levels = [blob_levels]
    if len(blob_levels) == 1:
        blob_levels = [blob_levels[0]] * 2
    assert len(blob_levels) == 2
    image = np.uint8((image >= min(blob_levels)) * (image <= max(blob_levels)) * 255)

    if erode_dilate_iters>0:
        eroded = cv2.erode(image, erode_dilate_kernel, iterations=erode_dilate_iters)
        if erode_dilate_connectivity in (4,8):
            num, labels = cv2.connectedComponents(eroded, connectivity=erode_dilate_connectivity)
            one_hot = (np.arange(1,num) == labels[..., np.newaxis]).astype(np.uint8)*255
        elif erode_dilate_connectivity==0:
            num = 2
            one_hot = eroded[..., np.newaxis]
        else:
            raise ValueError
        comps = [cv2.dilate(one_hot[..., i], erode_dilate_kernel, iterations=erode_dilate_iters) for i in range(num-1)]
    else:
        comps = [image]

    if approx_method is None:
        approx_method = cv2.CHAIN_APPROX_SIMPLE

    all_contours = []
    contours = []

    for comp in comps:
        zoom = cv2.resize(comp, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)   # this is in order to get a scalable vertex-following contour instead of a pixel-following contour

        _, comp_all_contours, _ = cv2.findContours(zoom, mode=cv2.RETR_LIST, method=approx_method)
        _, comp_contours, _ = cv2.findContours(zoom, mode=cv2.RETR_EXTERNAL, method=approx_method)

        all_contours += comp_all_contours
        contours += comp_contours

    if len(contours) != len(all_contours) and verbose:
        print('Warning: found and ignored %d child contours' % (len(all_contours) - len(contours)))

    svg = []
    if color is None:
        r = lambda: np.random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
    if label is None:
        label = color

    contours.sort(key=lambda x: str(x))
    skipped_small = 0
    for i in range(len(contours)):
        contours[i] = (contours[i] + 1) // 2

        if abs_eps or rel_eps:
            if simp_method == 'RDP':
                contours[i] = cv2.approxPolyDP(contours[i], max(abs_eps, rel_eps * cv2.arcLength(contours[i], closed=True)), closed=True)
            elif simp_method == 'VW':
                contours[i] = vw_closed(contours[i], max(abs_eps, rel_eps * cv2.contourArea(contours[i])))
            else:
                raise ValueError

        if cv2.contourArea(contours[i]) < min_area:
            skipped_small += 1
            contours[i] = None
            continue

        if box:
            contours[i] = cv2.boxPoints(cv2.minAreaRect(contours[i]))[:, None, :]

        # remove consecutive duplicates
        contour = [x[0][0] for x in groupby(contours[i].tolist())]
        if len(contour) > 1 and contour[-1] == contour[0]:
            del contour[-1]

        if len(contour)<3:
            skipped_small += 1
            contours[i] = None
            continue

        points = ' '.join(str(float(p[0])).rstrip('0').rstrip('.')+','+str(float(p[1])).rstrip('0').rstrip('.') for p in contour)
        svg.append('<polygon class="%s" fill="%s" id="%d" points="%s"/>' % (label, color, i, points))

    if skipped_small and verbose:
        print('Skipped %d small contours'%(skipped_small))

    if save_to is not None:
        save_svg(save_to, svg, resolution=image.shape[::-1])
        if save_png:
            svg2png(save_to, bg_color=png_bg_color)

    if show:
        print('%.1f sec' % (time() - start_time))
        cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cimage = cv2.copyMakeBorder(cimage, 0, 1, 0, 1, borderType=cv2.BORDER_CONSTANT)
        cv2.drawContours(cimage, [np.int0(c) for c in contours if c is not None], -1, color=(0, 0, 255))
        cv2.imshow('Found Contours', cimage)
        cv2.waitKey()

    return svg


def save_svg(filename, svg, resolution=None):
    if resolution is None:
        resolution = ''
        print('Warning: resolution not specified')
    else:
        resolution = 'width="%s" height="%s"' % (resolution[0], resolution[1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" version="1.1" %s>' % (resolution)] + svg + ['</svg>']
    with open(filename, 'w') as f:
        f.writelines(s + '\n' for s in svg)


def svg2png(filename, bg_color=None):
    from wand.image import Image, Color
    with Image(filename=filename) as image:
        if bg_color is not None:
            image.background_color = Color(bg_color)
            image.alpha_channel = 'remove'
        image.save(filename=filename + '.png')

if __name__ == '__main__':
    svg = blob2svg(image='test.png', save_to='test.svg', show=True)
