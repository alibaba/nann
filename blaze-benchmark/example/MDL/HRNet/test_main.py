from pose_hrnet import get_pose_net as get_hrnet
from pose_hrnet import model_cfg as hrnet_cfg
import time
import torch


if __name__ == '__main__':
    model = get_hrnet(hrnet_cfg, False, hdfs_client=None)
    model.cuda()
    model.eval()

    with torch.no_grad():
      inputs = torch.ones(1, 3, 800, 800).cuda()
      cnt = 20
  
      start = time.time()
      for _ in range(cnt):
          a = model.forward(inputs)
          a.cpu()
      end = time.time()
      print('avg time:{}'.format((end-start)/cnt))

