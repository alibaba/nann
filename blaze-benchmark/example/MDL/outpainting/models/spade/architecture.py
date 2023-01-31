import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class ResnetBlockWithSN(nn.Module):
    def __init__(self, fin, fout, opt, stride=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout) or (stride != 1)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, stride=stride)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, stride=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, stride=stride)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if 'syncbatch' in opt.norm_G:
            self.norm_0 = nn.SyncBatchNorm(fin, affine=True)
            self.norm_1 = nn.SyncBatchNorm(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.SyncBatchNorm(fin, affine=True)
        elif 'instance' in opt.norm_G:
            self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
            self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.InstanceNorm2d(fin, affine=True)
        else:
            self.norm_0 = nn.BatchNorm2d(fin, affine=True)
            self.norm_1 = nn.BatchNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.BatchNorm2d(fin, affine=True)

    def forward(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x)))
        else:
            x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
