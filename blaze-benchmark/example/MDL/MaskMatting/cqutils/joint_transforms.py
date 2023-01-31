import cv2
import numpy as np
import random
import collections
# from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask=None):
        if isinstance(img, list):
            if random.random() < 0.5:
                img = [cv2.flip(item, 1) for item in img]
                if mask is not None:
                    mask = [cv2.flip(item, 1) for item in mask]
            return img, mask
        else:
            if random.random() < 0.5:
                return img[:, ::-1, :], mask[:, ::-1]
            return img, mask

class RandomVerticallyFlip(object):
    def __call__(self, img, mask=None):
        if isinstance(img, list):
            if random.random() < 0.5:
                img = [item[::-1] for item in img]
                if mask is not None:
                    mask = [item[::-1] for item in mask]
            return img, mask
        else:
            if random.random() < 0.5:
                return img[::-1, :, :], mask[::-1, :]
            return img, mask

class ElasticTransform(object):
    def __init__(self):
        pass

    def __call__(self, img, mask):
        im_merge = np.concatenate((img, np.expand_dims(mask, axis=2)), axis=2)
        im_merge_t = self.elastic_transform(im_merge, im_merge.shape[1] * 2,
                                            im_merge.shape[1] * 0.08,
                                            im_merge.shape[1] * 0.02)
        img = im_merge_t[..., :3]
        mask = im_merge_t[..., 3]
        return img, mask

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

class Scale(object):
    def __init__(self, max_size, gt_resize_type=cv2.INTER_NEAREST):
        self.max_size = max_size
        self.gt_resize_type = gt_resize_type

    def __call__(self, img, mask=None):
        if isinstance(img, list):
            h, w = img[0].shape[:2]
            if (w <= h and w == self.max_size) or (h <= w and h == self.max_size):
                return img, mask
            if w < h:
                ow = self.max_size
                oh = int(self.max_size * h / w)
            else:
                oh = self.max_size
                ow = int(self.max_size * w / h)
            img = [cv2.resize(item, (ow, oh), interpolation=cv2.INTER_LINEAR)
                   for item in img]
            if mask is not None:
                mask = [cv2.resize(lab, (ow, oh), interpolation=self.gt_resize_type)
                        for lab in mask]
            return img, mask
        else:
            assert img.shape[:2] == mask.shape
            h, w = img.shape[:2]
            if (w <= h and w == self.max_size) or (h <= w and h == self.max_size):
                return img, mask
            if w < h:
                ow = self.max_size
                oh = int(self.max_size * h / w)
            else:
                oh = self.max_size
                ow = int(self.max_size * w / h)
            img, mask = cv2.resize(img, tuple([ow, oh])), cv2.resize(mask,
                    tuple([ow, oh]), interpolation=self.gt_resize_type)
            # print('Scale w={},h={}'.format(ow, oh))
            # print('Scale mask > 0 :{}'.format(mask[np.where(
            #     (mask !=0) & (mask !=1) & (mask != 2))]))
            return img, mask

class Resize(object):
    def __init__(self, size, gt_resize_type=cv2.INTER_NEAREST):
        self.size = size
        self.gt_resize_type = gt_resize_type

    def __call__(self, img, mask=None):
        if isinstance(img, list):
            if mask is None:
                new_size = tuple([self.size, self.size])
                img = [cv2.resize(item, new_size) for item in img]
            else:
                new_size = tuple([self.size, self.size])
                img = [cv2.resize(item, new_size) for item in img]
                mask = [cv2.resize(item, new_size,
                                   interpolation=self.gt_resize_type)
                        for item in mask]
        else:
            assert img.shape[:2] == mask.shape
            img = cv2.resize(img, tuple([self.size, self.size]))
            mask = cv2.resize(mask, tuple([self.size, self.size]),
                              interpolation=self.gt_resize_type)
        return img, mask

class RandomCrop(object):
    def __init__(self, crop_size, ignore_label=255, no_padding=False,
                 gt_resize_type=cv2.INTER_NEAREST):
        self._h = crop_size
        self._w = crop_size
        self._ignore_label = ignore_label
        self.no_padding = no_padding
        self.gt_resize_type = gt_resize_type

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        if self.no_padding:
            if w < self._w or h < self._h:
                if w < h:
                    ow = self._w
                    oh = int(self._w * h / w)
                else:
                    oh = self._h
                    ow = int(self._h * w / h)
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (ow, oh), interpolation=self.gt_resize_type)
        else:
            pad_h = max(self._h - h, 0)
            pad_w = max(self._w - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                         constant_values=((122,112), (104,104), (116,116)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)),
                          'constant', constant_values=self._ignore_label)
        return img, mask

    def __call__(self, img, mask=None):
        if isinstance(img, list):
            h, w = img[0].shape[: 2]
            if self.no_padding:
                if w < self._w or h < self._h:
                    if w < h:
                        ow = self._w
                        oh = int(self._w * h / w)
                    else:
                        oh = self._h
                        ow = int(self._h * w / h)
                    img = [cv2.resize(item, (ow, oh), interpolation=cv2.INTER_LINEAR)
                           for item in img]
                    if mask is not None:
                        mask = [cv2.resize(item, (ow, oh), interpolation=self.gt_resize_type)
                                for item in mask]
            else:
                pad_h = max(self._h - h, 0)
                pad_w = max(self._w - w, 0)
                img = [np.pad(item, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                              constant_values=((122,112), (104,104), (116,116)))
                       for item in img]
                if mask is not None:
                    mask = [np.pad(item, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                                  constant_values=self._ignore_label)
                            for item in mask]
            h, w = img[0].shape[: 2]
            x1 = random.randint(0, w - self._w)
            y1 = random.randint(0, h - self._h)
            img = [item[y1:y1+self._h,x1:x1+self._w] for item in img]
            if mask is not None:
                mask = [item[y1:y1 + self._h, x1:x1 + self._w] for item in mask]
            return img, mask
        else:
            assert img.shape[:2] == mask.shape

            img, mask = self._pad(img, mask)
            h, w = img.shape[:2]

            x1 = random.randint(0, w - self._w)
            y1 = random.randint(0, h - self._h)

            return img[y1:y1+self._h,x1:x1+self._w], mask[y1:y1+self._h,x1:x1+self._w]

class PFACrop(object):
    def __init__(self):
        pass

    def __call__(self, img, mask=None):
        if isinstance(img, list):
            h, w = img[0].shape[: 2]
            randh = np.random.randint(h / 8)
            randw = np.random.randint(w / 8)
            offseth = 0 if randh == 0 else np.random.randint(randh)
            offsetw = 0 if randw == 0 else np.random.randint(randw)
            p0, p1, p2, p3 = offseth, h + offseth - randh, offsetw, w + offsetw - randw

            img = [item[p0:p1, p2:p3] for item in img]
            if mask is not None:
                mask = [item[p0:p1, p2:p3] for item in mask]
            return img, mask
        else:
            assert img.shape[:2] == mask.shape
            h, w = img.shape[:2]
            randh = np.random.randint(h / 8)
            randw = np.random.randint(w / 8)
            offseth = 0 if randh == 0 else np.random.randint(randh)
            offsetw = 0 if randw == 0 else np.random.randint(randw)
            p0, p1, p2, p3 = offseth, h + offseth - randh, offsetw, w + offsetw - randw

            return img[p0:p1, p2:p3], mask[p0:p1, p2:p3]

class Padding(object):
    def __init__(self, ignore_label=255):
        super(Padding).__init__()
        self._ignore_label = ignore_label

    def __call__(self, img, mask=None):
        assert img.shape[:2] == mask.shape
        h, w = img.shape[:2]
        size = max(h, w)

        paddingh = (size - h) // 2
        paddingw = (size - w) // 2
        img = np.pad(img, ((0, paddingh), (0, paddingw), (0, 0)), 'constant',
                     constant_values=((122, 112), (104, 104), (116, 116)))
        mask = np.pad(mask, ((0, paddingh), (0, paddingw)),
                      'constant', constant_values=self._ignore_label)
        return img, mask

class RandomSized(object):
    def __init__(self, scale=[0.8, 1.2],
                 gt_resize_type=cv2.INTER_NEAREST):
        self.scale = scale
        self.gt_resize_type = gt_resize_type

    def __call__(self, img, mask):
        assert img.shape[:2] == mask.shape
        r = random.uniform(self.scale[0], self.scale[1])
        w = int(r * img.shape[1])
        h = int(r * img.shape[0])

        # img = cv2.resize(img,tuple([w,h]))
        # mask = cv2.resize(mask,tuple([w,h]))

        img, mask = cv2.resize(img, tuple([w, h])), cv2.resize(mask,
                tuple([w, h]), interpolation=self.gt_resize_type)
        
        # print('RandomSized w={},h={}'.format(w, h))
        # print('RandomSized mask > 0 :{}'.format(mask[np.where(
        #     (mask !=0) & (mask !=1) & (mask != 2))]))
        return img, mask

class RandomRotate(object):
    def __init__(self, degree, ignore_label=255, gt_resize_type=cv2.INTER_NEAREST):
        self.degree = degree
        self.ignore_label = ignore_label
        self.gt_resize_type = gt_resize_type

    def __call__(self, img, mask):
        assert img.shape[:2] == mask.shape

        h, w = img.shape[:2]
        center = tuple([int(w/2), int(h/2)])
        rotate_degree = random.random() * 2 * self.degree - self.degree
        M = cv2.getRotationMatrix2D(center, rotate_degree, 1.0)

        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=self.gt_resize_type)

        mask_reserve = 255*np.ones(mask.shape, np.uint8)
        mask_reserve = cv2.warpAffine(
                mask_reserve, M, (w, h), flags=cv2.INTER_NEAREST)
        mask[mask_reserve<255] = self.ignore_label
        return img, mask

class Apply(object):
    def __init__(self, transform, th):
        self.transform = transform
        self.th = th

    def __call__(self, *img):
        return self._apply_all(img, self.transform, self.th)

    def _apply_all(self, img, process_fn, th):
        if not isinstance(process_fn, collections.Sequence):
            img = [process_fn(item) if th[idx] else item
                   for idx, item in enumerate(img)]
        else:
            img = [process_fn[idx](item) if th[idx] else item
                   for idx, item in enumerate(img)]
        return img

if __name__ == '__main__':
    tmp = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/TB2aYztX2BNTKJjy1zdXXaScpXa_!!2261171538_g5_2YpAnnJA_m_cbmsqgld.jpg.jpg')
    mask = cv2.imread('D:/project/res_vis/alpha_fg_bg_comp/TB2aYztX2BNTKJjy1zdXXaScpXa_!!2261171538_g5_2YpAnnJA_m_cbmsqgld.jpg_gt.png',
                        cv2.IMREAD_GRAYSCALE)

    trans = PFACrop()
    tmp, mask = trans(tmp, mask)

    cv2.imshow('1', tmp)
    cv2.imshow('2', mask)
    cv2.waitKey()
