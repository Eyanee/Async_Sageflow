import torch
import numpy as np
import copy


# Lie attack
"""
input:
benign_user:良性用户的梯度

output:计算得到的恶意用户梯度
"""

def LIE_attack(benign_users):
        if len(benign_users) == 0:
            return

        users_grads = []
        for usr in benign_users:
            users_grads.append(usr.grads)

        grads_mean = np.mean(users_grads, axis=0) # 计算均值
        grads_stdev = np.var(users_grads, axis=0) ** 0.5  ## 计算标准差



        mal_grads = grads_mean + 0.74 * grads_stdev

        return mal_grads


# min-max:
"""
input:


output:
"""
def min_max(all_updates, dev_type='unit_vec'):
    model_re = np.mean(all_updates, axis=0) # 计算均值

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        

    threshold_diff = 1e-5
    base_grad = all_updates[0]
    for grad in all_updates:
        distance = torch.norm((grad - base_grad),axis = 0)**2
        max_distance = max(max_distance, distance)

    while torch.abs(lamda_succ - lamda) > threshold_diff: # 这个门槛值?不知道需不需要根据数据集去调

        mal_update = (model_re - lamda * deviation)
        for grad in all_updates:
            distance = torch.norm((grad - mal_update),axis = 0)**2
            max_d = max(max_d, distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

# min-sum
"""
input:
all_updates:所有良性更新

output: 
mal_update:恶意梯度

"""
def min_sum(all_updates, dev_type='unit_vec'):
    model_re = np.mean(all_updates, axis=0) # 计算均值

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        

    threshold_diff = 1e-5
    base_grad = all_updates[0]
    for grad_i in all_updates:
        for grad_j in all_updates:
            distance = torch.norm((grad - mal_update),axis = 0)**2
            sum_d_i += distance
        sum_d = max(sum_d, sum_d_i)

    while torch.abs(lamda_succ - lamda) > threshold_diff: # 这个门槛值?不知道需不需要根据数据集去调

        mal_update = (model_re - lamda * deviation)
        sumd_d_mal = 0
        for grad in all_updates:
            distance = torch.norm((grad - mal_update),axis = 0)**2
            sum_d_mal = sumd_d_mal + distance
        if sum_d_mal <= sum_d:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update



def Grad_tailored(all_updates, model_re, n_attackers, dev_type='unit_vec'):

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        
    lamda = torch.Tensor([20.0]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

def multi_krum(all_updates, n_attackers, multi_k=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0]])
        all_indices = np.delete(all_indices, indices[0])
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)
