import torch
import math

class CustomDistance1(torch.nn.Module):
    def __init__(self):
        super(CustomDistance1, self).__init__()

    def forward(self, ori_params, mod_params, ref_distance):

        for idx, key in enumerate(ori_params.keys()):
            if idx == 0:
                squared_sum = torch.sum(torch.pow(ori_params[key].requires_grad_(False)- mod_params[key].requires_grad_(True), 2))
            else:
                squared_sum += torch.sum(torch.pow(ori_params[key].requires_grad_(False) - mod_params[key].requires_grad_(True), 2))

        # 计算平方和的平方根，即模型参数之间的距离
        distance = squared_sum**(0.5) 
        return distance