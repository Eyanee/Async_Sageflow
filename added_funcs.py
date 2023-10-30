import copy
import math
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np
import torch.nn.functional as F

'''
做一个预处理，返回预处理完的值
'''

'''
攻击算法，mean，median，trimean下目标是最大化avg与mal update之间的loss
Krum和AFA的目标是恶意用户被选中
'''
    
def poison_Mean(para_updates, avg_update, args, user_number ,benign_user_number): #para_updates是所有用户的梯度；avg_update是聚合梯度，取para_updates的平均结果
    # avg_update = torch.mean(para_updates, 0)
    # n_attackers = args.num_users - args.benign_user_number
    dev_type = args.dev_type

    if dev_type == 'sign':
        deviation = torch.sign(avg_update)
    elif dev_type == 'unit_vec':
        deviation = avg_update / torch.norm(avg_update)  # unit vector, dir opp to good dir
    elif dev_type == 'std':
        deviation = torch.std(para_updates, 0)

    lamda = torch.Tensor([args.threshold]).cuda(0)  # compute_lambda_our(all_updates, model_re, n_attackers) #args.threshold, type=float, default=20.0
    prev_loss = -1
    step = lamda / 2
    lamda_succ = 0

    # print("user_number is ", user_number)
    # print("para_updates length is ", para_updates.size())
    i = 0
    while torch.abs(lamda_succ - lamda) > args.threshold_diff: #threshold_diff, type=float, default=1e-5
        mal_update = avg_update - lamda * deviation  #恶意梯度等于平均良性梯度减去偏差
        for client_num in range(user_number):
            if client_num >= benign_user_number:  #将恶意用户的梯度替换成上面构造的恶意梯度，这里是选取最后几个用户作为恶意用户
                para_updates[client_num] = mal_update.clone().detach() 

        agg_grads = grads_Mean(para_updates, args) # 每次重构之后的新的聚合梯度

        loss = torch.norm(agg_grads - avg_update) #构造的恶意梯度要使投毒后的聚合梯度与原本的聚合梯度之间距离尽可能大

        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + step / 2
        else:
            lamda = lamda - step / 2

        step = step / 2
        prev_loss = loss
        i = i + 1
    mal_update = avg_update - lamda_succ * deviation

    return mal_update

def grads_Mean(para_updates, args):
    mean_grads = torch.mean(para_updates, 0) # 计算平均值
    return mean_grads


def scale_attack(global_grad, ori_grad, scale_weight, current_number_of_adversaries):
   

    clip_rate = (scale_weight/ current_number_of_adversaries) * -1
    print(f"Scaling by  {clip_rate}")
    mod_grad = copy.deepcopy(ori_grad)

    for key in mod_grad.keys(): ## model.state_dict()

        target_value = mod_grad[key]
        value = global_grad[key]
        new_value = target_value + (value - target_value) * clip_rate 

        mod_grad[key].copy_(new_value)
        
    distance = model_dist_norm(ori_grad, mod_grad)
    # print(f" distance is  {distance}")
    return mod_grad


def scale_attack_mod(args, model, train_dataset, global_grad, ori_grad, current_number_of_adversaries):
    """
    对梯度进行反方向放缩，约束条件：
    1. 不被熵检测出来
    2. 放缩后的梯度与原梯度相比差尽可能大
    """
    scale_weight = args.scale_weight
    clip_rate = (scale_weight/ current_number_of_adversaries)  ### clip_rate可以修改成跟修改参数相关的因素的值  attacker_num/ staleness/
    print(f"Scaling by  {clip_rate}")
    mod_grad = copy.deepcopy(ori_grad)

    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    testloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    batch_entropy = []

    avg_batch_entropy = 2
    ## 在现在的位置做一个二分查找
    while avg_batch_entropy >= 1 or avg_batch_entropy <= 0.75:
        if avg_batch_entropy < 0.75:
            clip_rate = clip_rate * 2
        else:
            clip_rate = clip_rate/2
        print(f"step scaling by  {clip_rate}")
        new_grad = modifyGradient(clip_rate, ori_grad, global_grad)
        batch_entropy = []
        model.load_state_dict(new_grad)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            output, out = model(images)
            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            entropy  = -1.0 * Information.sum(dim=1) # size [64]
            average_entropy = entropy.mean().item()
            batch_entropy.append(average_entropy)
  
        
        avg_batch_entropy = sum(batch_entropy)/len(batch_entropy)

        

    return new_grad


def modifyGradient(clip_rate, ori_grad, global_grad):
    mod_grad = copy.deepcopy(ori_grad)
    for key in mod_grad.keys(): ## model.state_dict()

        target_value = mod_grad[key]
        value = global_grad[key]
        new_value = target_value + (value - target_value) * clip_rate 

        mod_grad[key].copy_(new_value)
    return mod_grad

def model_dist_norm(ori_params, mod_params):
    squared_sum = 0


    # 遍历参数字典，计算差异的平方和
    for name in ori_params:
        squared_sum += torch.sum(torch.pow(ori_params[name] - mod_params[name], 2))

    # 计算平方和的平方根，即模型参数之间的距离
    distance = math.sqrt(squared_sum)
    return distance



    
    # # 对dataset中的label做反转

    # # return 

