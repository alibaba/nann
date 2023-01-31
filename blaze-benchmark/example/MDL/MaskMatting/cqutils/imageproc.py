import cv2
import numpy as np
import logging
import sys
import os
import math
import random
import traceback
sys.path.append(os.path.dirname(__file__))
# print('sys.path:{}'.format(sys.path))
from tfs.client import TfsClient
from solve_foreground_background import solve_foreground_background
import urllib.request
import urllib
# import mmh3 as mmh
import uuid
import requests

from media_center_io import MediaCenterIO


__all__ = [
        'scale_image',
        'estimate_foreground',
        'load_tfs',
        'entropy1d',
        'entropy2d',
        'image_synthesis',
        ]

SCALE_MODE = ['shorter', 'longer', 'factor']

tfs_client = TfsClient(app_key='52413f88bcefe',
        root_server='restful-store.vip.tbsite.net:3800',
        logger=logging.getLogger())


mc_client = MediaCenterIO('0', 'scs')

#rule1 = LifecycleRule('rule1', 'qingmu/',
#                      status=LifecycleRule.ENABLED,
#                      expiration=LifecycleExpiration(days=365))
#
#lifecycle = BucketLifecycle([rule1])
#oss_bucket.put_bucket_lifecycle(lifecycle)


def _resize_image_on_shorter_dim(img, tdim, ip):
    h, w = img.shape[:2]
    if min(h, w) <=  tdim:
        return img
    if h < w:
        th = tdim
        ratio = float(th) / h
        tw = int(ratio * w)
    else:
        tw = tdim
        ratio = float(tw) / w
        th = int(ratio * h)
    img = cv2.resize(img, (tw, th), interpolation=ip)
    return img

def _resize_image_on_longer_dim(img, tdim, ip):
    h, w = img.shape[:2]
    if max(h, w) <= tdim:
        return img
    if h > w:
        th = tdim
        ratio = float(th) / h
        tw = int(ratio * w)
    else:
        tw = tdim
        ratio = float(tw) / w
        th = int(ratio * h)
    img = cv2.resize(img, (tw, th), interpolation=ip)
    return img

def _resize_image_on_ratio(img, factor, ip):
    if factor == 1:
        return img
    h, w = img.shape[:2]
    th = int(factor * h)
    tw = int(factor * w)
    img = cv2.resize(img, (tw, th), interpolation=ip)
    return img

def scale_image(img, mode, arg, ip=cv2.INTER_LINEAR):
    if mode not in SCALE_MODE:
        raise RuntimeError('mode:{} is not supported yet. Canidcates are {}'.format(mode, SCALE_MODE))
    if mode == SCALE_MODE[0]:  # 'shorter'
        return _resize_image_on_shorter_dim(img, arg, ip)
    if mode == SCALE_MODE[1]:  # 'longer'
        return _resize_image_on_longer_dim(img, arg, ip)
    if mode == SCALE_MODE[2]:  # factor
        return _resize_image_on_ratio(img, arg, ip)

# def read_image_from_url(url, flags=cv2.IMREAD_COLOR):
#     try:
#         tfs = url.strip().split('/')[-1]
#         url = os.path.join('https://img.alicdn.com/tfscom', tfs)
#         img_data = urllib.request.urlopen(url).read()
#         img_data = np.asarray(bytearray(img_data), dtype='uint8')
#         img = cv2.imdecode(img_data, flags)
#         return img
#     except Exception as e:
#         try:
#             img_data = urllib.request.urlopen(url).read()
#             img_data = np.asarray(bytearray(img_data), dtype='uint8')
#             img = cv2.imdecode(img_data, flags)
#             return img
#         except Exception as e:
#             print('Read image from {} failed.'.format(url))


def read_tfs(tfs, flags=cv2.IMREAD_COLOR, intranet=True):
    tfs = tfs.strip()
    img = None
    try:
        if intranet:
            tfs = tfs.split('/')[-1]
            # url = os.path.join('https://img.alicdn.com/tfscom', tfs)
            # img_content = urllib.request.urlopen(url).read()
            img_content = tfs_client.read_file(tfs)
        else:
            #img_content = urllib.request.urlopen(tfs).read()
            res = requests.get(tfs, allow_redirects=False)
            if res.status_code in [301, 302, 303, 307, 308]:
                print("redirected url!")
                return None
            img_content = res.content
        img_data = np.asarray(bytearray(img_content), dtype='uint8')
        img = cv2.imdecode(img_data, flags)
    except Exception as e:
        print('Read {} failed.'.format(tfs))
        traceback.print_exc()
    return img

def load_tfs(tfs, intranet=True):
    # load tfs image byte data without decode
    tfs = tfs.strip()
    img_data = None
    try:
        if intranet:
            tfs = tfs.split('/')[-1]
            url = os.path.join('https://img.alicdn.com/tfscom', tfs)
            img_content = urllib.request.urlopen(url).read()
            # img_content = tfs_client.read_file(tfs)
        else:
            #img_content = urllib.request.urlopen(tfs).read()
            headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
            res = requests.get(tfs, allow_redirects=False, headers=headers)
            if res.status_code in [301, 302, 303, 307, 308]:
                print("redirected url!")
                return None
            img_content = res.content
        img_data = np.asarray(bytearray(img_content), dtype='uint8').tobytes()
    except Exception as e:
        print('Read {} failed.'.format(tfs))
        traceback.print_exc()
    return img_data


def write_tfs(img, suffix, intranet=True):
    tfs = None
    try:
        if intranet:
            img_content = cv2.imencode(suffix, img)[1]
        else:
            logging.info('Only support intranet!')
            img_content = None
        img_data = np.array(img_content).tostring()
        #print(img_data[:100])
        #print(len(img_data))
        tfs = mc_client.put_data(img_data)
        # tfs = tfs_client.write_file(img_data, suffix=suffix)
    except Exception as e:
        logging.info('Write image to tfs failed.')
        #print(e)
        raise e
    return tfs

def get_uid_by_url(url):
        #return mmh.hash64(filename)[1]
    res = uuid.uuid3(uuid.NAMESPACE_URL, url)
    return str(res)


def estimate_foreground(img, alpha, closed_form=False):
    alpha_norm = alpha / 255.0
    if closed_form:
        img_norm = img / 255.0
        est_fg, est_bg = solve_foreground_background(img_norm, alpha_norm)
        est_fg = (est_fg * 255.0).astype(np.uint8)
    else:
        est_fg = img
    alpha_norm_channels = np.expand_dims(alpha_norm, 2)
    alpha_norm_channels = np.repeat(alpha_norm_channels, 3, 2)
    act_fg = (alpha_norm_channels * est_fg).astype(np.uint8)
    act_fg = np.concatenate((act_fg, np.expand_dims(alpha, 2)), axis=2)
    return act_fg

def entropy1d(image_gray):
    h, w = image_gray.shape[:2]
    tmp = np.bincount(image_gray.flatten())

    tmp = tmp / (h*w)
    res = 0
    for i in range(len(tmp)):
        if tmp[i] != 0:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))

    return res

def entropy2d(image_gray, patch_size=5):
    h, w = image_gray.shape[:2]
    center = patch_size // 2

    kernel = np.ones((patch_size, patch_size))
    kernel[center, center] = 0
    kernel /= np.sum(kernel)

    dst = cv2.filter2D(image_gray, -1, kernel)
    dst = dst.astype(np.uint8)

    statistics = np.zeros((256, 256))
    image_gray = image_gray.flatten()
    dst = dst.flatten()
    for src_item, dst_item in zip(image_gray, dst):
        statistics[src_item, dst_item] += 1
    statistics /= (h * w)
    statistics = statistics.flatten()

    res = 0
    for i in range(len(statistics)):
        if statistics[i] != 0:
            res = float(res - statistics[i] * (math.log(statistics[i]) / math.log(2.0)))
    return res

def image_synthesis(fg_img, bg_img, size):
    def preprocess(img, size, mode='max'):
        h, w = img.shape[:2]
        if mode == 'max':
            if h > w:
                new_w = size
                new_h = int((size / w) * h)
            else:
                new_h = size
                new_w = int((size / h) * w)
        else:
            if h < w:
                new_w = size
                new_h = int((size / w) * h)
            else:
                new_h = size
                new_w = int((size / h) * w)
        img = cv2.resize(img, (new_w, new_h))
        return img

    bg_img = preprocess(bg_img, size)
    bg_h, bg_w = bg_img.shape[:2]

    start_x = int(random.random() * bg_w/2)
    start_y = int(random.random() * bg_h/2)
    delta_x = random.randint(int((bg_w-start_x)*0.9), bg_w - start_x)
    delta_y = random.randint(int((bg_h-start_y)*0.9), bg_h - start_y)
    fg_img = preprocess(fg_img, min(delta_x, delta_y), mode='min')

    fg_h, fg_w = fg_img.shape[:2]
    alpha = fg_img[:, :, 3].copy()
    alpha = np.tile(np.expand_dims(alpha, 2), (1, 1, 3))/255.0
    bg_img[start_y:start_y+fg_h, start_x:start_x+fg_w, :] = \
        alpha*fg_img[:, :, :3] + (1-alpha)*bg_img[start_y:start_y+fg_h, start_x:start_x+fg_w, :]
    bg_img = bg_img.astype(np.uint8)

    new_alpha = np.zeros((bg_h, bg_w))
    new_alpha[start_y:start_y + fg_h, start_x:start_x + fg_w] = fg_img[:, :, 3]
    return bg_img, alpha
