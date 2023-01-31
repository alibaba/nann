import torch
import os
import cv2
import numpy as np
import requests
from models.spade.generator import OutpaintingGenerator
import nvtx
from argparse import Namespace

class NetGModle():
  def init_opt(self):
    return Namespace(
        use_gpu=True,
        batchSize=1,
        downsample_first_layer=True,
        use_unet=True,
        isTrain=False,
        outpainting=True,
        no_seg=True,
        semantic_nc=3 + 1,
        label_nc=0,
        norm_G='spectralspadebatch3x3',
        ngf=64,
        init_type='xavier',
        init_variance=0.02,
        no_spade=True,
        crop_size=512,
        times=4,
        aspect_ratio=1,
        debug=False
    )

  def __init__(self):
    opt = self.init_opt()
    opt.segmentation = False
    self.netG = OutpaintingGenerator(opt)
    self.netG.eval()
    self.netG.init_weights(opt.init_type, opt.init_variance)

    assert (torch.cuda.is_available())
    self.netG.cuda()

    self.h, self.w = opt.crop_size, opt.crop_size
    self.r = opt.times

    mask = np.zeros((self.h, self.w), dtype=np.int32)
    mask[:, -self.w // self.r:] = 1
    mask = mask[:, :, np.newaxis]
    mask = mask.astype("f").transpose(2, 0, 1)
    self.mask_tensor = torch.from_numpy(mask).float().unsqueeze(dim=0)
    self.mask_tensor = self.mask_tensor.cuda()

  def forward(self, input_image):
    input_image = input_image.float()
    fake_image = self.netG(input_image.unsqueeze(dim=0), None, self.mask_tensor)
    fake_image = fake_image[0].permute(1, 2, 0)
    fake_image = (fake_image + 1) / 2.0 * 255.0
    return fake_image.to(torch.uint8)


