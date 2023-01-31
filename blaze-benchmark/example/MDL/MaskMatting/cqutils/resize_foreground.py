# coding: utf-8

from __future__ import division
import numpy as np
import cv2
import glob
import json
import os
import mxnet
import mtcnn_detector

'''
is_the_image_big_enough?
  |--yes:tag==waist_or_tag==leg?
           |--yes:adjust_the_bounding_box
           |--no:is_there_a_face?
                   |--yes:is_the_face_big_enough?
                            |--yes:detect_blank
                            |--no:adjust_the_bounding_box
                   |--no:tag==shoulder?
                            |--yes:adjust_the_bounding_box
                            |--no:is_the_ratio_good_enough?
                                    |--yes:detect_blank
                                    |--no:adjust_the_bounding_box         
         detect_blank:
           |--detected_blank_on_top/bottom?
                |--yes:cut_the_blank
                |--no:do_not_crop
         adjust_the_bounding_box:
           |--(show_the_hat_or_shoes)
           |--be_a_square
           |--do_not_reach_beyond_the_picture
           |--(do_not_add_contours)
           |--(no_pixel_on_the_left/right_edge)
  |--no:do_not_crop
'''
'''
swith(tag)
    case head: miny<=top_boundary
    case shoulder: show the head and place the shoulder at the center
    case waist: put the waist at the center
    case leg: only hips&knees (prioritized) or knees&ankles are needed,
              and place the thigh (or calf) at the center
    case foot: maxy>=bottom_boundary
    case else: default
'''

def detect_face(png, detector):
    bgr = cv2.merge(cv2.split(png)[0:3])
    results = detector.detect_face(bgr)
    # print 'detect done'
    if results is not None:
        face_box = results[0][0]
        points = results[1][0]
        print "face:", face_box
        # print "points:", points
        return [face_box, points]
    else:
        return None


def row_is_transparent(alpha, row):
    # return alpha[row].nonzero()[0].size<10
    return alpha[row].sum()<200


def col_is_transparent(alpha, col):
    # return alpha[:,col].nonzero()[0].size<10
    return alpha[:,col].sum()<200


def find_bounding_box(alpha):
    h,w = alpha.shape
    flag_num = 5

    flag = flag_num # to deal with noise
    left_boundary = 0
    while flag>0:
        if not col_is_transparent(alpha, left_boundary):
            flag -= 1
        else:
            flag = flag_num
        left_boundary += 1
    left_boundary -= flag_num

    flag = flag_num
    right_boundary = w-1
    while flag>0:
        if not col_is_transparent(alpha, right_boundary):
            flag -= 1
        else:
            flag = flag_num
        right_boundary -= 1
    right_boundary += flag_num

    flag = flag_num
    top_boundary = 0
    while flag>0:
        if not row_is_transparent(alpha, top_boundary):
            flag -= 1
        else:
            flag = flag_num
        top_boundary += 1
    top_boundary -= flag_num

    flag = flag_num
    bottom_boundary = h-1
    while flag>0:
        if not row_is_transparent(alpha, bottom_boundary):
            flag -= 1
        else:
            flag = flag_num
        bottom_boundary -= 1
    bottom_boundary += flag_num

    return [top_boundary, bottom_boundary, left_boundary, right_boundary]


# new_size is not the final size, new_size can be float, and can be smaller than space_size
def adjust_box(center_y, center_x, new_size, h, w, space_size):
    final_size = int(min(new_size,h,w))
    final_size = max(space_size,final_size)
    delta = final_size - 1
    miny = int(max(center_y-final_size/2.0, 0))
    maxy = int(min(miny+delta, h-1))
    miny = int(max(maxy-delta, 0))
    minx = int(max(center_x-final_size/2.0, 0))
    maxx = int(min(minx+delta, w-1))
    minx = int(max(maxx-delta, 0))
    return [miny, maxy, minx, maxx]


def count_contours(gray):
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in contours:
        if len(c)>20:
            cnt += 1
    return cnt


def advanced_adjust_box(center_y, center_x, new_size, h, w, space_size, alpha):
    cnt = count_contours(alpha)
    # print "cnt:", cnt
    step1 = 100
    miny, maxy, minx, maxx = [0, 0, 0, 0]

    while True:
        miny, maxy, minx, maxx = adjust_box(center_y, center_x, new_size, h, w, space_size)

        # no pixels on the left & right edges
        step2 = 5
        while maxx<=w-1:
            if col_is_transparent(alpha, maxx):
                break
            maxx += step2
        maxx = min(maxx,w-1)
        while minx>=0:
            if col_is_transparent(alpha, minx):
                break
            minx -= step2
        minx = max(minx, 0, maxx-h+1)
        delta = maxx - minx
        maxy = min(miny+delta,h-1)
        miny = max(maxy-delta,0)
        
        if maxy-miny>=h-1 or maxx-minx>=w-1:
            break
        new_cnt = count_contours(alpha[miny:maxy+1,minx:maxx+1])
        # print "new_cnt:", new_cnt
        if new_cnt<=cnt:
            break
        print "enlarge the box"
        new_size += step1

    return [miny, maxy, minx, maxx]


def crop_blank(foreground, bounding_box, space_size, space_ratio):
    print "begin to crop (blank)"
    h, w, _ = foreground.shape

    top_boundary, bottom_boundary, left_boundary, right_boundary = bounding_box
    item_w = right_boundary - left_boundary + 1
    item_h = bottom_boundary - top_boundary + 1
    size = max(item_h, item_w, space_size)
    max_space = int( size/(1.0-2.0*space_ratio)*space_ratio )

    top_boundary = max(top_boundary-max_space, 0)
    bottom_boundary = min(bottom_boundary+max_space, h-1)
    left_boundary = max(left_boundary-max_space, 0)
    right_boundary = min(right_boundary+max_space, w-1)
    item_w = right_boundary - left_boundary + 1
    item_h = bottom_boundary - top_boundary + 1
    center_x = (left_boundary + right_boundary) / 2.0
    center_y = (top_boundary + bottom_boundary) / 2.0
    new_size = max(item_h, item_w, space_size)
    miny, maxy, minx, maxx = adjust_box(center_y, center_x, new_size, h, w, space_size)

    new_foreground = foreground[miny:maxy+1,minx:maxx+1]
    return new_foreground


def get_tag_from_info(info):
    if info is not None and len(info)==3:
        tag, center, half_len = info
        if not (tag is None or tag=="else"):
            if not (tag=="shoulder" and center is None):
                if not (tag=="waist" and center is None):
                    if not (tag=="leg" and (center is None or half_len is None)):
                        return tag
    return "else"


def crop_foreground(foreground, space_size, min_face_h, detector=None, info=None):
    h, w, _ = foreground.shape
    tag = get_tag_from_info(info)
    ideal_ratio = 1.618
    space_ratio = 0.0625

    if h<=space_size or w<=space_size:
        print "the image is too small, no need to crop"
        return foreground

    alpha = cv2.split(foreground)[3]
    kernel = np.ones((5,5),np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    bounding_box = find_bounding_box(alpha)
    print "bounding_box:", bounding_box
    top_boundary, bottom_boundary, left_boundary, right_boundary = bounding_box
    item_w = right_boundary - left_boundary + 1
    item_h = bottom_boundary - top_boundary + 1

    if tag=="waist":
        print "begin to crop (waist)"
        center_x, center_y = info[1]
        new_size = min(1.25*item_w, h, w)
        miny, maxy, minx, maxx = advanced_adjust_box(center_y, center_x, new_size, h, w, space_size, alpha)
        return foreground[miny:maxy+1, minx:maxx+1]
    elif tag=="leg":
        print "begin to crop (leg)"
        center_x, center_y = info[1]
        new_size = min(info[2]*2.0, h, w)
        miny, maxy, minx, maxx = advanced_adjust_box(center_y, center_x, new_size, h, w, space_size, alpha)
        return foreground[miny:maxy+1, minx:maxx+1]

    if detector is None:
        detector = mtcnn_detector.MtcnnDetector(model_folder='model', ctx=mxnet.gpu(0), num_worker = 1, accurate_landmark = False)
        print 'init done'
    result = detect_face(foreground,detector)

    if result is not None and result[0][-1]>0.92:
        print "detected a face"

        face_minx, face_miny, face_maxx, face_maxy, _ = result[0]
        face_w = face_maxx - face_minx
        face_h = face_maxy - face_miny
        center_x = ( face_maxx + face_minx ) / 2.0
    
        if face_h/item_h<min_face_h/space_size:
            print "begin to crop (face)"

            if face_h<=min_face_h or h<=space_size or w<=space_size:
                scale = 1.0
            else:
                scale = min(face_h/min_face_h, h/space_size, w/space_size)

            new_size = space_size * scale
            if tag=="head":
                top = top_boundary - space_ratio * new_size
            else:
                top = face_miny - face_h/2.0
            center_y = top + new_size/2.0
            miny, maxy, minx, maxx = advanced_adjust_box(center_y, center_x, new_size, h, w, space_size, alpha)

            # print scale,miny,maxy,minx,maxx
            return foreground[miny:maxy+1, minx:maxx+1]  

        else:
            print "the face is big enough"

    else:
        print "no face in the foreground"

        if tag=="shoulder": # no face, but still a person
            print "begin to crop (shoulder)"
            center_x, center_y = info[1]

            new_size = (center_y-top_boundary) / (1.0-space_ratio) * 2.0
            miny, maxy, minx, maxx = advanced_adjust_box(center_y, center_x, new_size, h, w, space_size, alpha)
            return foreground[miny:maxy+1, minx:maxx+1]

        if item_h/item_w>ideal_ratio: # ignore the situation when item_w/item_h>ideal_ratio
            print "begin to crop (object)"
            new_size = int(max( (right_boundary-left_boundary)*ideal_ratio, space_size ))
            center_x = (right_boundary+left_boundary) / 2.0
            if tag=="foot":
                bottom = min(bottom_boundary+new_size*space_ratio, h-1)
                center_y = bottom - new_size/2.0
            elif tag=="head":
                top = max(top_boundary-new_size*space_ratio, 0)
                center_y = top + new_size/2.0
            else:
                center_y = h/2.0
            miny, maxy, minx, maxx = adjust_box(center_y, center_x, new_size, h, w, space_size)
            # print miny,maxy,minx,maxx
            return foreground[miny:maxy+1, minx:maxx+1]
        
        else:
            print "the ratio is acceptable (object)"

    new_foreground = crop_blank(foreground, bounding_box, space_size, space_ratio)
    return new_foreground


# example: batch_crop_foreground('pretreat_pose_result.json', '../matting_results/', 'crop/')
def batch_crop_foreground(info_file, input_dir, output_dir, num=-1):
    if (not os.path.exists(info_file)) or (not os.path.exists(input_dir)) or (not os.path.exists(output_dir)):
        print "INPUT ERROR"
        return

    detector = mtcnn_detector.MtcnnDetector(model_folder='model', ctx=mxnet.gpu(0), num_worker = 1, accurate_landmark = False)
    print 'init done'

    with open(info_file, "r") as f:
        tags, centers, half_lens = json.load(f)

    foreground_names = glob.glob(input_dir+'/*.png')
    for index, foreground_name in enumerate(foreground_names):
        tfs = os.path.splitext(os.path.basename(foreground_name))[0]
        foreground = cv2.imread(foreground_name, -1) # flag<0 means that the alpha channel is needed
        
        if tfs not in tags:
            info_file = None
            tag = "else"
        if info_file is not None:
            tag = tags[tfs]
            if tfs in centers:
                center = centers[tfs]
            else:
                center = None
            if tfs in half_lens:
                half_len = half_lens[tfs]
            else:
                half_len = None
            info = [tag,center,half_len]
        else:
            info = None
            
        new_foreground = crop_foreground(foreground, space_size=200, min_face_h=70, detector=detector, info=info)
        cv2.imwrite(output_dir+"/"+tfs+'.png', new_foreground)
        print 'No.{} {} done'.format(index+1,tfs)
        if num>0 and index+1>=num:
            break

