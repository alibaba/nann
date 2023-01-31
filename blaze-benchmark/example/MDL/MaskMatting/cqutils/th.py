import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import cv2

__all__ = [
        'initialize_weights',
        'CrossEntropyLoss2d',
        'vis_tensor_to_trimap',
        'vis_trimap',
        'vis_saliency_segment',
        'dss_net_output_non_binary',
        'str2tensor',
        'tensor2str',
        'print_network',
        'denormalize_to_images',
        'dense_crf',
        'crop_3_2',
        'crop_scale',
        'max_domain',
        'matting_postprocess',
        ]

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in',
                #         nonlinearity='relu')
                nn.init.kaiming_normal(module.weight, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

def vis_tensor_to_trimap(tensor, softmax):
    """visualize a C x H x W tensor where C=3 to a a trimap"""
    trimap_list = []
    if softmax:
        tensor = F.softmax(tensor, dim=1)
    for i in range(tensor.size(0)):
        trimap = torch.squeeze(tensor[i, :, :, :].cpu().data).numpy() * 255
        trimap = trimap.transpose(1, 2, 0).astype(np.uint8)
        trimap_list.append(trimap)
    return trimap_list

def denormalize_to_images(tensor, mean, std):
    """visualize a B x C x H x W normlized image tensor where C=3 to an image"""
    image_list = []
    for i in range(tensor.size(0)):
        ts = torch.squeeze(tensor[i, :, :, :])
        for t, m, s in zip(ts, mean, std):
            t.mul_(s).add_(m)
        image = ts.cpu().numpy() * 255
        image = np.transpose(image, [1, 2, 0]).astype(np.uint8)
        image = image[:, :, ::-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list

def vis_trimap(tensor):
    trimap_list = []
    for i in range(tensor.size(0)):
        trimap_compact = torch.squeeze(tensor[i, :, :].cpu().data).numpy()
        trimap = np.zeros(trimap_compact.shape, dtype=np.uint8)
        trimap[np.where(trimap_compact == 0)] = 128
        trimap[np.where(trimap_compact == 1)] = 255
        trimap[np.where(trimap_compact == 2)] = 0
        trimap_list.append(trimap)
    return trimap_list

def dss_net_output_non_binary(tensor):
    res_list = []
    for i in range(tensor.size(0)):
        seg_compact = torch.squeeze(tensor[i, :, :, :].cpu().data).numpy()
        seg_compact *= 255
        res_list.append(seg_compact)
    return res_list

def vis_saliency_segment(tensor, threshold=0):
    res_list = []
    for i in range(tensor.size(0)):
        seg_compact = torch.squeeze(tensor[i, :, :, :].cpu().data).numpy()
        seg_compact *= 255
        seg = np.zeros(seg_compact.shape, dtype=np.uint8)
        seg[np.where(seg_compact > threshold)] = 255
        res_list.append(seg)
    return res_list

def str2tensor(s):
    """
    Convert asscci string to int tensor
    """
    t = torch.IntTensor(len(s))
    for i, c in enumerate(s):
        t[i] = ord(c)
    return t
def tensor2str(t):
    """
    Convert int tensor to asscci string
    """
    s = ''
    for i in range(t.size(0)):
        s += chr(t[i])
    return s

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

def dense_crf(img, non_binary_output, M=2):
    import pydensecrf.densecrf as dcrf
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    tau = 1.05
    EPSILON = 1e-8
    anno_norm = non_binary_output
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    # Do the inference
    res = np.argmax(d.inference(5), axis=0)

    res = res.reshape(img.shape[:2])
    return res

def crop_3_2(img, pos):
    org_size = img.shape
    x_0, y_0, x_1, y_1 = pos
    x_0, y_0, x_1, y_1 = int(x_0), int(y_0), int(x_1), int(y_1)

    x_cneter = (x_0 + x_1) // 2
    y_cneter = (y_0 + y_1) // 2

    x_dist = min(x_cneter, org_size[1] - x_cneter)
    y_dist = min(y_cneter, org_size[0] - y_cneter)

    scale = y_dist / x_dist
    if scale < 1.5:
        x_dist = ceil(2*y_dist/3)
    else:
        y_dist = ceil(x_dist*1.5)
    new_x_0, new_x_1, new_y_0, new_y_1 = \
        x_cneter-x_dist, x_cneter+x_dist, y_cneter-y_dist, y_cneter+y_dist
    new_w, new_h = new_x_1-new_x_0, new_y_1-new_y_0
    img_crop = img[new_y_0:new_y_1, new_x_0:new_x_1]

    bbox_area = (x_1 - x_0) * (y_1 - y_0)
    intersection_area = (min(x_1, new_x_1) - max(x_0, new_x_0)) * \
                        (min(y_1, new_y_1) - max(y_0, new_y_0))
    recall = intersection_area / bbox_area

    # if new_w<(x_1-x_0) or new_h<(y_1-y_0):
    #     return img_crop, recall
    if new_h < 600:
        return None, recall

    return img_crop, recall

def crop_scale(img, pos, target_scale, is_resize=False):
    target_height, target_width = target_scale
    target_ratio = target_height / target_width
    org_size = img.shape
    x_0, y_0, x_1, y_1 = pos
    x_0, y_0, x_1, y_1 = int(x_0), int(y_0), int(x_1), int(y_1)

    x_cneter = (x_0 + x_1) // 2
    y_cneter = (y_0 + y_1) // 2

    x_dist = min(x_cneter, org_size[1] - x_cneter)
    y_dist = min(y_cneter, org_size[0] - y_cneter)

    scale = y_dist / x_dist
    if scale < target_ratio:
        x_dist = ceil(y_dist / target_ratio)
    else:
        y_dist = ceil(x_dist * target_ratio)
    new_x_0, new_x_1, new_y_0, new_y_1 = \
        x_cneter-x_dist, x_cneter+x_dist, y_cneter-y_dist, y_cneter+y_dist
    new_w, new_h = new_x_1-new_x_0, new_y_1-new_y_0
    img_crop = img[new_y_0:new_y_1, new_x_0:new_x_1]

    bbox_area = (x_1 - x_0) * (y_1 - y_0)
    intersection_area = (min(x_1, new_x_1) - max(x_0, new_x_0)) * \
                        (min(y_1, new_y_1) - max(y_0, new_y_0))
    recall = intersection_area / bbox_area
    crop_area = new_w * new_h
    precision = intersection_area / crop_area

    if is_resize:
        img_crop = cv2.resize(img_crop, (target_width, target_height))
        ratio = target_width / new_w
        return img_crop, recall, precision, ratio
    else:
        # if new_h < 200:
        #     return None, recall, precision
        return img_crop, recall, precision, 1

def max_domain(img_gray):

    # find contours of all the components and holes
    gray_temp = np.zeros_like(img_gray)  # copy the gray image because function
    gray_temp[img_gray > 0] = 255
    if gray_temp.sum() == 0:
        return np.zeros_like(img_gray)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_temp)
    label_idx = np.bincount(labels.flatten())[1:].argmax()+1
    img_res = np.zeros_like(img_gray)
    img_res[labels == label_idx] = 255

    # findContours will change the input image into another
    # _, contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # show the contours of the imput image
    # cv2.drawContours(img_gray, contours, -1, (0, 255, 255), 2)

    # find the max area of all the contours and fill it with 0
    # area = []
    # for i in range(len(contours)):
    #     area.append(cv2.contourArea(contours[i]))
    # max_idx = np.argmax(area)
    # res = np.zeros_like(img_gray)
    # print(contours[max_idx])
    # cv2.fillConvexPoly(res, contours[max_idx], 255)
    return img_res

def matting_postprocess(alpha):
    '''
        remove small area
    '''
    mask = np.zeros_like(alpha)
    mask[alpha > 10] = 255
    if mask.sum() == 0:
        return alpha

    retval, labels, stats, centroids =\
        cv2.connectedComponentsWithStats(mask)

    max_domain_area = 0
    for state in stats[1:]:
        if max_domain_area < state[-1]:
            max_domain_area = state[-1]
    if max_domain_area == 0:
        total_num = alpha.shape[0] * alpha.shape[1]
        max_domain_area = total_num
    for idx, state in enumerate(stats):
        _, _, _, _, num = state
        if num < (max_domain_area * 0.5):
            alpha[labels == idx] = 0
    return alpha

if __name__ == '__main__':
    # import torchvision
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # model = torchvision.models.resnet50()
    # model = torchvision.models.resnet101()
    # # print([name for name, param in model.named_parameters()])
    # model = model.to(device)
    # print_network(model)

    # def read_url(url, method=cv2.IMREAD_COLOR):
    #     import urllib.request
    #     resp = urllib.request.urlopen(url, timeout=1)
    #     img = np.asarray(bytearray(resp.read()), dtype="uint8")
    #     img = cv2.imdecode(img, method)
    #     return img
    #
    # img = read_url('https://img.alicdn.com/bao/uploaded/i1/113911071/O1CN01XBDI4g1JmYi4M9iX1_!!0-saturn_solar.jpg')
    # pos = np.array([61.38822937011719, 0.9156951904296875, 238.8485565185547, 327.37628173828125])
    # org_size = img.shape
    # scaling_factor = min(320 / org_size[1], 1)
    # pos[[0, 2]] += (320 - scaling_factor * org_size[1]) / 2
    # pos[[1, 3]] += (320 - scaling_factor * org_size[0]) / 2
    # new_w = int(org_size[1] * min(320 / org_size[1], 320 / org_size[0]))
    # new_h = int(org_size[0] * min(320 / org_size[1], 320 / org_size[0]))
    # pos[[0, 2]] -= (320 - new_w) // 2
    # pos[[1, 3]] -= (320 - new_h) // 2
    # scale = org_size[1] / new_w
    # pos[[0, 2]] *= scale
    # pos[[1, 3]] *= scale
    # img_crop, recall = crop_3_2(img, pos)
    # if img_crop is not None:
    #     cv2.imshow('1', img)
    #     cv2.imshow('2', img_crop)
    #     cv2.waitKey()

    img = cv2.imread('D:/project/res_vis/HRMNet/test/10_raw.png',
                     cv2.IMREAD_GRAYSCALE)
    img = matting_postprocess(img)
    cv2.imshow('2', img)
    cv2.waitKey()
