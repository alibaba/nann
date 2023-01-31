from model import TASED
import time
import torch

model = TASED()
model.cuda()

if __name__ == '__main__':
    #snippet = torch.cat(snippet, dim=-1)
    #snippet = img
    #snippet = snippet.permute(2, 0, 1).float()
    #inputs = snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(0, 2, 1, 3, 4)
    inputs = torch.ones(1,3,32,224,384).half().cuda()
    cnt = 20

    start = time.time()
    for _ in range(cnt):
        a = (model(inputs))
        a.cpu()
    end = time.time()
    print('avg time:{}'.format((end-start)/cnt))

