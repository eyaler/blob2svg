import numpy as np
import cv2
from itertools import groupby
from time import time
import re
from xml.sax.saxutils import quoteattr
import xml.etree.ElementTree as ET

def rgb2hex(color):
    return '#%02x%02x%02x' % tuple(color)

def hex2rgb(color):
    return tuple(int(color.strip().lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

def get_area(contour, i):
    p1 = contour[i - 1, 0]
    p2 = contour[i, 0]
    p3 = contour[i + 1 if i + 1 < len(contour) else 0, 0]
    return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2


def vw_closed(contour, area):
    areas = [get_area(contour, i) for i in range(len(contour))]
    while len(contour) > 2:
        min_area = min(areas)
        if min_area > area:
            break
        i = np.argmin(areas)
        contour = np.delete(contour, i, axis=0)
        del areas[i]
        if len(areas):
            areas[i - 1] = get_area(contour, i - 1)
            j = i if i < len(areas) else 0
            areas[j] = get_area(contour, j)
    return contour


def polybreak(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return [], []

    internal_dist = 0
    for i in range(len(defects)):
        s, e, f, d = defects[i, 0]
        if d >= internal_dist:
            internal_dist = d
            start_ind = s
            end_ind = e
            internal_ind = f
            internal_point = contour[f, 0]

    cut_dist = float('inf')
    i = (end_ind + 1) % len(contour)
    while i != start_ind:
        d = np.linalg.norm(internal_point - contour[i, 0])
        if d < cut_dist:
            cut_dist = d
            external_ind = i
        i = (i + 1) % len(contour)
    if cut_dist == float('inf'):
        return [], []

    cyclic = np.concatenate([contour] * 2)
    contour1 = cyclic[external_ind:internal_ind + 1 + len(contour) * (internal_ind < external_ind)]
    contour2 = cyclic[internal_ind:external_ind + 1 + len(contour) * (external_ind < internal_ind)]
    return [contour1, contour2], [tuple(internal_point), tuple(contour[external_ind, 0])]

def draw_exact_contours(image, contours, color):
    mask = np.zeros((image.shape[0]*2, image.shape[1]*2))
    contours = [contour * 2 for contour in contours]
    cv2.fillPoly(mask, contours, 255)
    mask = cv2.erode(mask, np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8))
    mask = cv2.resize(mask, dsize=None, fx=0.5, fy=0.5)
    image[mask>0] = color
    return image

def blob2svg(image, blob_levels=(1, 255), approx_method=None, simp_method='VW', abs_eps=0, rel_eps=0, min_area=0,
             box=False, box_min_frac=0, erode_dilate_iters=0, erode_dilate_kernel=np.ones((3, 3)),
             erode_dilate_connectivity=8, label=None, color=None, random_colors=True,
             save_to=None, bg_color=None, save_png=False, use_wand=False, show=0, verbose=True):
    # approx_method can be one of: cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE (or None, default), cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS
    # simp_method can be one of: 'RDP', 'VW'
    # connectivity can be one of: 0, 4, 8

    start_time = time()
    np.random.seed(0)

    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif len(image.shape)==3:
        image = np.mean(image, axis=-1)

    try:
        len(blob_levels)
    except:
        blob_levels = [blob_levels]
    if len(blob_levels) == 1:
        blob_levels = [blob_levels[0]] * 2
    assert len(blob_levels) == 2
    image = np.uint8((image >= min(blob_levels)) * (image <= max(blob_levels)) * 255)

    if erode_dilate_iters:
        eroded = cv2.erode(image, erode_dilate_kernel, iterations=erode_dilate_iters)
        if erode_dilate_connectivity in (4, 8):
            num, labels = cv2.connectedComponents(eroded, connectivity=erode_dilate_connectivity)
            one_hot = (np.arange(1, num) == labels[..., np.newaxis]).astype(np.uint8) * 255
        elif erode_dilate_connectivity == 0:
            num = 2
            one_hot = eroded[..., np.newaxis]
        else:
            raise ValueError
        comps = [cv2.dilate(one_hot[..., i], erode_dilate_kernel, iterations=erode_dilate_iters) for i in
                 range(num - 1)]
    else:
        comps = [image]

    if approx_method is None:
        approx_method = cv2.CHAIN_APPROX_SIMPLE

    all_contours = []
    contours = []

    for comp in comps:
        zoom = cv2.resize(comp, dsize=None, fx=2, fy=2,
                          interpolation=cv2.INTER_NEAREST)  # this is in order to get a scalable vertex-following contour instead of a pixel-following contour

        _, comp_all_contours, _ = cv2.findContours(zoom, mode=cv2.RETR_LIST, method=approx_method)
        _, comp_contours, _ = cv2.findContours(zoom, mode=cv2.RETR_EXTERNAL, method=approx_method)

        all_contours += comp_all_contours
        contours += comp_contours

    if len(contours) != len(all_contours) and verbose:
        print('Warning: found and ignored %d child contours' % (len(all_contours) - len(contours)))

    svg = []
    if color is None and random_colors:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    if isinstance(color, (tuple,list,np.ndarray)):
        color = rgb2hex(color)
    if label is None and color is not None:
        label = color

    contours.sort(key=lambda x: str(x))
    break_points = []
    skipped_small = 0
    broke_boxes = 0
    i = -1
    original_contours = len(contours)
    while i < len(contours) - 1:
        i += 1
        if i < original_contours:
            contours[i] = (contours[i] + 1) // 2

        # remove consecutive duplicates
        contours[i] = np.array([x[0] for x in groupby(contours[i].tolist())])
        if len(contours[i]) > 1 and np.array_equal(contours[i][-1], contours[i][0]):
            contours[i] = contours[i][:-1]

        if abs_eps or rel_eps:
            if simp_method == 'RDP':
                contours[i] = cv2.approxPolyDP(contours[i],
                                               max(abs_eps, rel_eps * cv2.arcLength(contours[i], closed=True)),
                                               closed=True)
            elif simp_method == 'VW':
                contours[i] = vw_closed(contours[i], max(abs_eps, rel_eps * cv2.contourArea(contours[i])))
            else:
                raise ValueError

        area = cv2.contourArea(contours[i])

        if area < min_area:
            skipped_small += 1
            contours[i] = None
            continue

        if box:
            box_contour = cv2.boxPoints(cv2.minAreaRect(contours[i]))[:, None, :]
            if area < box_min_frac * cv2.contourArea(box_contour):
                sub_contours, bps = polybreak(contours[i])
                if len(sub_contours):
                    broke_boxes += 1
                    contours += sub_contours
                    break_points.append(bps)
                    contours[i] = None
                    continue
            contours[i] = box_contour

        if len(contours[i]) < 3:
            skipped_small += 1
            contours[i] = None
            continue

        points = ' '.join(
            str(float(p[0, 0])).rstrip('0').rstrip('.') + ',' + str(float(p[0, 1])).rstrip('0').rstrip('.') for p in
            contours[i])

        cls = ''
        if label is not None:
            cls = ' class=%s'%quoteattr(label)
        if color is not None:
            fill = ' fill="%s"'%color
        else:
            fill = ' fill-opacity="0"'
        polygon = '<polygon%s%s id="%d" points="%s"/>' % (cls, fill, i, points)
        svg.append(polygon)

    if verbose:
        if skipped_small:
            print('Skipped %d small contours' % (skipped_small))
        if broke_boxes:
            print('Broke %d boxes' % (broke_boxes))

    if save_to is not None:
        save_svg(save_to, svg, dimensions=image.shape[::-1], bg_color=bg_color, save_png=save_png, use_wand=use_wand)

    if show:
        print('%.1f sec' % (time() - start_time))
        cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cimage = cv2.copyMakeBorder(cimage, 0, 1, 0, 1, borderType=cv2.BORDER_CONSTANT)
        if show<=2:
            contours = [np.int0(c) for c in contours if c is not None]
            cv2.drawContours(cimage, contours, -1, (0, 0, 255))
            if show == 2:
                for bps in break_points:
                    cv2.circle(cimage, bps[0], 4, (0, 255, 0), thickness=2)
                    cv2.circle(cimage, bps[1], 4, (255, 0, 0), thickness=2)
        elif show==3:
            for c in contours:
                if c is not None:
                    color = (np.random.randint(64, 256), np.random.randint(64, 256), np.random.randint(64, 256))
                    draw_exact_contours(cimage, [np.int0(c)], color)
        cv2.imshow('Found Contours', cimage)
        cv2.waitKey()

    return svg


def save_svg(filename, svg, dimensions=None, bg_color=None, save_png=False, use_wand=False):
    if dimensions is None:
        dimensions = ''
        print('Warning: dimensions not specified')
    else:
        dimensions = ' width="%s" height="%s"' % (dimensions[0], dimensions[1])
    if bg_color is None:
        bg_color = ''
    else:
        if isinstance(bg_color, (tuple, list)):
            bg_color = rgb2hex(bg_color)
        bg_color = ' style="background-color: %s"' % bg_color

    svg = ['<?xml version="1.0"?>']+['<svg xmlns="http://www.w3.org/2000/svg" version="1.1"%s%s>' % (dimensions, bg_color)] + svg + ['</svg>']
    with open(filename, 'w') as f:
        f.writelines(s + '\n' for s in svg)
    if save_png:
        svg2png(filename, use_wand=use_wand)

def svg2png(svg_filename_or_string, png_filename=None, use_wand=False, limit_labels=[]):
    assert not use_wand or len(limit_labels)==0
    blob = False
    try:
        root = ET.parse(svg_filename_or_string).getroot()
    except:
        root = ET.fromstring(svg_filename_or_string)
        blob = True
        assert png_filename is not None
    if png_filename is None:
        png_filename = svg_filename_or_string
    bg_color = None
    if 'style' in root.attrib:
        match = re.search('background-color: (#[A-Fa-f0-9]{6})', root.attrib['style'])
        if match:
            bg_color = match.group(1)
    if use_wand:
        from wand.image import Image, Color
        with Image(filename=svg_filename_or_string if not blob else None, blob=svg_filename_or_string if blob else None) as image:
            image.compression_quality = 9
            if bg_color is not None:
                image.background_color = Color(bg_color)
                image.alpha_channel = 'remove'
            image.save(filename=png_filename + '.png')
    else:
        if 'width' in root.attrib and 'height' in root.attrib:
            height = int(float(root.attrib['height']))
            width = int(float(root.attrib['width']))
        else:
            raise ValueError('Could not find height and width attributes in svg')
        image = np.zeros((height, width, 4), dtype=np.uint8)
        if bg_color is not None:
            image[:] = hex2rgb(bg_color)+(255,)
        for child in root:
            if not child.tag.endswith('polygon') or (len(limit_labels)>0 and 'class' in child.attrib and child.attrib['class'] not in limit_labels):
                continue
            if 'fill' in child.attrib:
                contour = np.array([[[float(point.split(',')[0]), float(point.split(',')[1])]] for point in child.attrib['points'].split(' ')], dtype=np.int0)
                draw_exact_contours(image, [contour], hex2rgb(child.attrib['fill'])+(255,))
        cv2.imwrite(png_filename + '.png', cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA), (cv2.IMWRITE_PNG_COMPRESSION, 9))

def test_svg2png(filename):
    blob2svg(image=filename, save_to=filename.replace('.png','.svg'), save_png=True, verbose=False)
    png1 = cv2.imread(filename.replace('.png','.svg.png'))
    blob2svg(image=filename, save_to=filename.replace('.png','_wand.svg'), save_png=True, use_wand=True, verbose=False)
    png2 = cv2.imread(filename.replace('.png', '_wand.svg.png'))
    cv2.imwrite(filename.replace('.png', '_diff.png'), np.uint8(np.abs(png1-png2)))
    bad_pixels = np.sum(~np.isclose(png1, png2))
    if bad_pixels>0:
        print('Warning: found %d bad pixels' % bad_pixels)
    else:
        print('svg2png passed test')

if __name__ == '__main__':
    #test_svg2png('test.png')
    svg = blob2svg(image='test.png', save_to='test.svg', show=2, save_png=True)
