from matting import MaskMattingModel
import time
import torch


if __name__ == '__main__':
    model = MaskMattingModel()

    with torch.no_grad():
      inputs = torch.ones(512, 512, 3).cuda()
      mask = torch.ones(512, 512).cuda()
      cnt = 20
  
      start = time.time()
      for _ in range(cnt):
          a = model.predict(inputs, mask)
          a.cpu()
      end = time.time()
      print('avg time:{}'.format((end-start)/cnt))

