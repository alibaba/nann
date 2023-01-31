from modules.modeling_no_grad import CLIP4Clip
from modules.task_config import TaskConfig as task_config
import time
import torch


if __name__ == '__main__':
    model = CLIP4Clip.from_pretrained("cross-base", 
            task_config=task_config, 
            clip_path='./modules/ViT-B-32.pt')
    model.cuda()

    input0 = torch.ones(1, 512).float().cuda()
    input1 = torch.ones(1, 512).float().cuda()
    input2 = torch.ones(20, 3, 224, 224).float().cuda()
    input3 = torch.ones(1, 20).int().cuda()
    cnt = 20

    start = time.time()
    for _ in range(cnt):
        a = model.get_sim_matrix_vta(input0, input1, input2, input3)
        a.cpu()
    end = time.time()
    print('avg time:{}'.format((end-start)/cnt))

