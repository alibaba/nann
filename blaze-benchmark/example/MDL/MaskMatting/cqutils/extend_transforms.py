import random
import cv2
import numpy as np
import collections
from skimage.filters import gaussian
import torch
from PIL import Image, ImageFilter

class MapToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.int64))

class ImageToTensor(object):
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, img):
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()

class ImageFormatToTensor(object):
    """HWC to CHW, normalize if specified
    input should be pytorch tensor, preferred on gpu already
    """
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, img):
        #img = img.cuda()
        print('image shape ', img.shape)
        img = img.permute((2,0,1))
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()



class GrayImageToTensor(object):
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, img):
        img = np.expand_dims(img, 2)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()

class GrayImageFormatToTensor(object):
    '''input should be pytorch tensor, preferred on gpu already'''
    def __init__(self, normalize=True):
        self.nomalize = normalize
    def __call__(self, img):
        #img = img.cuda()
        img = img.unsqueeze(2).permute((2, 0, 1))
        if self.nomalize:
            return img.float().div(255)
        else:
            return img.float()

class RandomGaussianBlur(object):
    def __init__(self, blur_prob):
        self.blur_prob = blur_prob

    def __call__(self, img):
        if random.random() < self.blur_prob:
            sigma = 0.15 + random.random() * 1.15
            blurred_img = gaussian(img, sigma=sigma, multichannel=True)
            blurred_img *= 255
            return blurred_img.astype(np.uint8)
        else:
            return img

class RandomBright(object):
    def __init__(self, p=0.5, lower=0.5, upper=1.5):
        super(RandomBright, self).__init__()
        self.p = p
        self.lower = lower
        self.upper = upper

    def __call__(self, img):
        if random.random() < self.p:
            mean = np.mean(img)
            img = img - mean
            img = img * random.uniform(self.lower, self.upper) \
                  + mean * random.uniform(self.lower, self.upper)
            img[img > 255] = 255
            img[img < 0] = 0
        return img

class RandomGamma(object):
    def __init__(self, gammas=[0.1,0.2,0.4,0.67,1,1.5,2.5,5]):
        super(RandomGamma, self).__init__()
        self.gammas = gammas

    def __call__(self, images):
        gv = random.choice(self.gammas)
        images = ((images.astype(np.float32) / 255.0) ** gv) * 255
        return images

class SingleScale(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h, w = img.shape[:2]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)
        img = cv2.resize(img, tuple([ow, oh]),
                         interpolation=self.interpolation)
        # print('Scale w={},h={}'.format(ow, oh))
        # print('Scale mask > 0 :{}'.format(mask[np.where(
        #     (mask !=0) & (mask !=1) & (mask != 2))]))
        return img

class SingleResize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = cv2.resize(img, tuple([self.size, self.size]),
                         interpolation=self.interpolation)
        return img

class SingleRandomCrop(object):
    def __init__(self, crop_size, no_padding=False,
                 interpolation=cv2.INTER_LINEAR):
        self._h = crop_size
        self._w = crop_size
        self.no_padding = no_padding
        self.interpolation = interpolation

    def _pad(self, img):
        h, w = img.shape[: 2]
        if self.no_padding:
            if w < self._w or h < self._h:
                if w < h:
                    ow = self._w
                    oh = int(self._w * h / w)
                else:
                    oh = self._h
                    ow = int(self._h * w / h)
                img = cv2.resize(img, (ow, oh), interpolation=self.interpolation)
        else:
            pad_h = max(self._h - h, 0)
            pad_w = max(self._w - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        return img

    def __call__(self, img):
        img = self._pad(img)
        h, w = img.shape[:2]

        x1 = random.randint(0, w - self._w)
        y1 = random.randint(0, h - self._h)

        return img[y1:y1+self._h,x1:x1+self._w,:]

class Padding(object):
    def __init__(self, ignore_label=255):
        super(Padding).__init__()
        self._ignore_label = ignore_label

    def __call__(self, img):
        h, w = img.shape[:2]
        size = max(h, w)

        paddingh = (size - h) // 2
        paddingw = (size - w) // 2
        img = np.pad(img, ((0, paddingh), (0, paddingw), (0, 0)), 'constant',
                     constant_values=(((122, 112), (104, 104), (116, 116))))
        return img

class SimpleNormalize(object):
    def __call__(self, img):
        return img.div_(255.0).mul_(2).add_(-1)

if __name__ == '__main__':
    tmp = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/TB2aYztX2BNTKJjy1zdXXaScpXa_!!2261171538_g5_2YpAnnJA_m_cbmsqgld.jpg.jpg')

    trans = Padding()
    tmp = trans(tmp)
    tmp = tmp.astype(np.uint8)
    cv2.imshow('', tmp)
    cv2.waitKey()
