import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torch.autograd import Function

from utils import initialize_weights

res50_path = '/models/resnet/resnet50-19c8e357.pth'

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=True):
        super(PSPNet50, self).__init__()
        self.use_aux = use_aux
        resnet = models.resnet50()
        if pretrained:
            import os.path as osp 
            if not osp.exists(res50_path):
                raise RuntimeError('Please ensure {} exists.'.format(res50_path))
            resnet.load_state_dict(torch.load(res50_path))
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')


class gradient_reversal_function(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class GradientReversal(nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()

    def forward(self, input):
        return gradient_reversal_function.apply(input)


class PSPNet50CoTrain(nn.Module):
    def __init__(self, num_shm_classes, num_coco_classes, pretrained=True, use_aux=True):
        super(PSPNet50CoTrain, self).__init__()
        self.use_aux = use_aux
        resnet = models.resnet50()
        if pretrained:
            import os.path as osp 
            if not osp.exists(res50_path):
                raise RuntimeError('Please ensure {} exists.'.format(res50_path))
            resnet.load_state_dict(torch.load(res50_path))
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) 
        )
        self.shm_classifier = nn.Conv2d(512, num_shm_classes, kernel_size=1)
        self.coco_classifier = nn.Conv2d(512, num_coco_classes, kernel_size=1)
        self.gradient_reversal_layer = GradientReversal()
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, 2, kernel_size=1)
        )

        if use_aux:
            self.shm_aux_logits = nn.Conv2d(1024, num_shm_classes, kernel_size=1)
            self.coco_aux_logits = nn.Conv2d(1024, num_coco_classes, kernel_size=1)
            initialize_weights(self.shm_aux_logits, self.coco_aux_logits)

        initialize_weights(self.ppm, self.final)
        initialize_weights(self.shm_classifier, self.coco_classifier)
        initialize_weights(self.domain_classifier)

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_aux:
            shm_aux = self.shm_aux_logits(x)
            shm_aux = F.upsample(shm_aux, x_size[2:], mode='bilinear')
            coco_aux = self.coco_aux_logits(x)
            coco_aux = F.upsample(coco_aux, x_size[2:], mode='bilinear')
        x = self.layer4(x)
        x = self.ppm(x)
        
        x_cls = self.final(x)
        shm_pred = self.shm_classifier(x_cls)
        shm_pred = F.upsample(shm_pred, x_size[2:], mode='bilinear')
        
        coco_pred = self.coco_classifier(x_cls)
        coco_pred = F.upsample(coco_pred, x_size[2:], mode='bilinear')

        x = self.gradient_reversal_layer(x)
        domain_pred = self.domain_classifier(x)
        domain_pred = F.upsample(domain_pred, x_size[2:], mode='bilinear')

        if self.use_aux:
            return shm_pred, shm_aux, coco_pred, coco_aux, domain_pred
        return shm_pred, coco_pred, domain_pred


if __name__=='__main__':
    from torch.autograd import Variable

    #model = PSPNet50(2, pretrained=False).cuda()
    #print(model)

    #x = Variable(torch.randn(2, 3, 800, 948).cuda())
    #y, aux = model.forward(x)

    #print('aux.size:')
    #print(aux.size())

    x = Variable(torch.randn(5,5).cuda(), requires_grad=True)
    gr = GradientReversal(2)
    y = gr(x).sum()

    y.backward()
    print(x)
    print(x.grad)
