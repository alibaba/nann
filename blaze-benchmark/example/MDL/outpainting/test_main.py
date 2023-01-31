from models.NetGModle import NetGModle
import time
import torch


if __name__ == '__main__':
    model = NetGModle()
    inputs = torch.ones(3, 512, 512).half().cuda()
    cnt = 20

    start = time.time()
    for _ in range(cnt):
        a = model.forward(inputs)
        a.cpu()
    end = time.time()
    print('avg time:{}'.format((end-start)/cnt))

