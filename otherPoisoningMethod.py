import torch
import numpy as np
import copy

def get_key_list(std_keys):
    res = list()
    for key in std_keys:
        res.append(key)

    return res

def modifyWeight(std_keys, local_weights):  
    """
    local_weights : state_dict 字典值
    """
    param_updates = list()
    param_update = list()
    for update_item in local_weights:
        param_new = []
        for key in std_keys:
            param_new.append(copy.deepcopy(update_item[key]))

        param_update = [] # 清空
        for j in range(len(param_new)):
            sub_res = torch.sub(param_new[j], 0).reshape(-1)
            param_update = sub_res if len(param_update) == 0 else torch.cat((param_update, sub_res), 0)

        param_updates = param_update.clone().unsqueeze(0) if len(param_updates) == 0 else torch.cat((param_updates, param_update.clone().unsqueeze(0)), dim=0)  # 先用unsqueeze(0)增加维度

    return param_updates


def restoreWeight(std_dict, std_keys, update_weights):
    # 重构张量，重构字典 
    update_dict = copy.deepcopy(std_dict)
    front_idx = 0
    end_idx = 0
    # mal_update张量重构

    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        # print("update_weights shape", type(update_weights))
        # print("front idx and end idx", front_idx, end_idx)
        tmp_tensor = update_weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = copy.deepcopy(tmp_tensor)
        front_idx = end_idx
    return update_dict



# Lie attack
"""
input:
benign_user:良性用户的梯度

output:计算得到的恶意用户梯度
"""

def LIE_attack(benign_users):
    if len(benign_users) == 0:
        print("zero")
        return
    std_dict = copy.deepcopy(benign_users[0])
    std_keys = get_key_list(std_dict.keys())
    users_grads = []
    for usr in benign_users:
        param = modifyWeight(std_keys, usr)
        users_grads.append(param)

    params_mean = torch.mean(users_grads, axis=0) # 计算均值
    params_stdev = torch.var(users_grads, axis=0) ** 0.5  ## 计算标准差


    print("params_mean", params_mean.shape)
    print("params_stdev", params_stdev.shape)
    
    mal_param = params_mean + 0.74 * params_stdev
    print("mal_param", mal_param.shape)
    mal_param = restoreWeight(std_dict, std_keys,mal_param)

    return mal_param


# min-max:
"""
input:


output:
"""
def min_max(all_updates, dev_type='std'):
    ### allupdates这边好像有问题
    std_keys = all_updates[0].keys()
    std_dict = copy.deepcopy(all_updates[0])
    param_updates = modifyWeight(std_keys, all_updates)
    model_re = np.mean(param_updates, axis=0) # 计算均值

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(param_updates, 0)
        

    threshold_diff = 1e-5
    base_grad = param_updates[0]
    for grad in param_updates:
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
    
    mal_update = restoreWeight(std_dict, std_keys, mal_update)

    return mal_update

# min-sum
"""
input:
all_updates:所有良性更新

output: 
mal_update:恶意梯度

"""
def min_sum(param_updates, dev_type='unit_vec'):

    std_dict = copy.deepcopy(param_updates.values[0])
    std_keys = get_key_list(std_dict.keys())
    all_updates = modifyWeight(std_keys, param_updates)

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
            distance = torch.norm((grad_i - grad_j),axis = 0)**2
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

    mal_update = restoreWeight(std_dict, std_keys, mal_update)

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




### LA attack
    ## on Trimmed Mean and Mean full knowledge
def LA_attack(all_updates, n_attackers):
    model_re = torch.mean(all_updates, 0)
    model_std = torch.std(all_updates, 0)
    deviation = torch.sign(model_re)
    
    max_vector_low = model_re + 3 * model_std 
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
    min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    return mal_vec


