import numpy as np
import cv2
import torch

__all__ = [
        'inverse_transform',
        'revise_inference_scale',
        'inverse_padding',
        ]

def inverse_transform(img_tensor, mean, std):
    """
    transform img_tensor to unscaled img
    input: 
        img_tensor - Num x Channel x Height x Width, tensor
        mean - Channel, list
        std - Channel, list
    output:
        img_list - a list including Num items,
                   each is a Height x Width x Channel array(rgb, np.uint8)
    """
    assert len(img_tensor.shape) == 4
    assert img_tensor.shape[1] == len(mean)
    assert img_tensor.shape[1] == len(std)

    for chn_id, (m_val, s_val) in enumerate(zip(mean, std)):
        img_tensor[:,chn_id,:,:] = img_tensor[:,chn_id,:,:]*s_val + m_val

    img_tensor = torch.clamp(img_tensor, 0, 1)
    img = img_tensor.cpu().numpy().transpose([0,2,3,1])*255
    return list(img.astype(np.uint8))

def revise_inference_scale(img_array, h, w, input_size, interpolation=cv2.INTER_NEAREST):
    new_h, new_w = img_array.shape[:2]
    if (w <= h and w == input_size) or (h <= w and h == input_size):
        return img_array[:h, :w]
    if w < h:
        ow = input_size
        oh = int(input_size * h / w)
        return cv2.resize(img_array[:oh, :ow], (w, h), interpolation=interpolation)
    else:
        oh = input_size
        ow = int(input_size * w / h)
        return cv2.resize(img_array[:oh, :ow], (w, h), interpolation=interpolation)

def inverse_padding(pridict, shape, interpolation=cv2.INTER_LINEAR):
    h, w = shape
    size = max(h, w)
    pridict = cv2.resize(pridict, (size,size),
                         interpolation=interpolation)
    paddingh = (size - h) // 2
    paddingw = (size - w) // 2
    return pridict[paddingh:h + paddingh, paddingw:w + paddingw]
