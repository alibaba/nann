# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
# import pycocotools.mask as mask_util


# Use a non-interactive backend
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

__all__ = [
        'vis_mask',
        'vis_bbox',
        ]

_WHITE = np.array([255, 255, 255])
def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_bbox(image, bboxs, label=None, score=None, label_names=None):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        bboxs (list): box data with the format of [x1, y1, x2, y2]
            in the coordinate system of the input image.
    """
    for i in range(len(label)):
        cat_size = cv2.getTextSize(label_names[int(label[i])] + '000',
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = (0, 255, 0)
        txt = '{}{:.0f}'.format(label_names[int(label[i])], score[i] * 100)
        bbox = bboxs[i]
        if bbox[1] - cat_size[1] - 2 < 0:
            cv2.rectangle(image,
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                          color, -1)
            cv2.putText(image, txt,
                        (bbox[0], bbox[1] + cat_size[1] + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        else:
            cv2.rectangle(image,
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2),
                          color, -1)
            cv2.putText(image, txt,
                        (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.rectangle(image,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      color, 2)
    return image
