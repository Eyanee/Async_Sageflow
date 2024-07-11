
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
    
    users_grads=modifyWeight(std_keys,benign_users)

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
def min_max(args, all_updates, dev_type='std'):
    ### allupdates这边好像有问题
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    std_keys = all_updates[0].keys()
    std_dict = copy.deepcopy(all_updates[0])
    print("len all_updates is ", len(all_updates))
    # 传参有问题
    param_updates = modifyWeight(std_keys, all_updates)
    print("param_updates type  is ", param_updates.shape)

    model_re = torch.mean(param_updates, axis=0) # 计算均值

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(param_updates, 0)
        
    max_distance = 0
    max_d = 0
    lamda_succ = lamda = torch.Tensor([11.0]).to(device)
    lamda = torch.Tensor([10.0]).to(device)
    lamda_fail = lamda
    threshold_diff = 1e-3
    # base_grad = param_updates[0]
    for grad_i in param_updates:
        for grad_j in param_updates:
            distance = torch.norm(grad_i - grad_j)**2
        # print("distance is ",distance)
        max_distance = max(max_distance, distance)
        # print("max distance is ",max_distance)
    

    while torch.abs(lamda_succ - lamda) > threshold_diff: # 这个门槛值?不知道需不需要根据数据集去调

        mal_update = (model_re - lamda * deviation)
        print("lamda is ",lamda)
        for grad in param_updates:
            distance = torch.norm(grad - mal_update)**2
            max_d = max(max_d, distance)
        if max_d <= max_distance:
            print("there")
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
            break
        else:
            print("here")
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
def min_sum(args, param_updates, dev_type='unit_vec'):
    
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    # print("len all_updates is ", len(param_updates))
    # all_updates = modifyWeight(std_keys, param_updates)
    all_updates = torch.stack(param_updates,dim = 0)

    model_re = torch.mean(all_updates, axis=0) # 计算均值s

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    
    lamda_succ = torch.Tensor([0]).to(device)
    lamda = torch.Tensor([50.0]).to(device)
    lamda_fail = lamda
    sum_d_i = 0
    sum_d = 0

    threshold_diff = 1e-5
    # base_grad = all_updates[0]
    for grad_i in all_updates:
        sum_d_i = 0
        for grad_j in all_updates:
            distance = torch.norm(grad_i - grad_j)**2
            sum_d_i += distance
        sum_d = max(sum_d, sum_d_i)

    while torch.abs(lamda_succ - lamda) > threshold_diff: # 这个门槛值?不知道需不需要根据数据集去调

        mal_update = (model_re - lamda * deviation)
        sumd_d_mal = 0
        for grad in all_updates:
            distance = torch.norm(grad - mal_update)**2
            sum_d_mal = sumd_d_mal + distance
        if sum_d_mal <= sum_d:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
            # break
            # print("there")
        else:
            lamda = lamda - lamda_fail / 2
            # print("here")

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update



def Grad_median(args, param_updates, n_attackers, dev_type='unit_vec', threshold=5.0):
    ## model_re是params的均值 
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')
    
    # std_dict = copy.deepcopy(param_updates[0])
    # std_keys = get_key_list(std_dict.keys())
    # all_updates = modifyWeight(std_keys, param_updates)
    all_updates = torch.stack(param_updates,dim = 0)

    model_re = torch.mean(all_updates, axis=0) # 计算均值



    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(param_updates, 0)

    lamda = torch.Tensor([threshold]).to(device)#compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        # print("shape param_updates ",param_updates.shape)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = torch.median(mal_updates, 0)[0]
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

def compute_gradient(model_1, model_2, std_keys,  lr):
    grad = list()
    for key in std_keys:
        param1 = model_1[key]
        param2 = model_2[key]
    #     # 根据公式计算梯度
        # tmp = (param1 - param2)tmp = (param1 - param2)/lr.view(-1)
        tmp = (param1 - param2)
        # tmp = (param1 - param2)
        grad = tmp.view(-1) if len(grad)== 0 else torch.cat((grad,tmp.view(-1)),0)
    print("grad,shape is",grad.shape)
    return grad

def restoregradients(std_dict, std_keys, update_weights):
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
        update_dict[k] = copy.deepcopy(tmp_tensor) +  update_dict[k]
        front_idx = end_idx
    return update_dict

def getAllGraidients(args, params,global_model,std_keys):
    grads = list()
    for param in params:
        grad  = compute_gradient(param,global_model.state_dict(),std_keys,args.lr)
        grads.append(grad)
    grads =torch.stack(grads,dim  = 0)

    return grads

### LA attack
    ## on Trimmed Mean and Mean full knowledge
def modifyLA(std_keys, param_updates,global_model,lr):
    res_list = []
    for param in param_updates:
        param_mod = compute_gradient(param,global_model.state_dict(),std_keys,lr)
        res_list.append(param_mod)
    return torch.stack(res_list,dim = 0)
        
    
def LA_attack(args, param_updates, n_attackers,global_model,std_keys):

    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    # std_dict = copy.deepcopy(param_updates[0])
    # std_keys = get_key_list(std_dict.keys())
    # all_updates = getAllGraidients(args, param_updates,global_model,std_keys)

    # std_dict = copy.deepcopy(param_updates[0])
    # all_updates = modifyWeight(std_keys, param_updates)
    all_updates = torch.stack(param_updates,dim = 0)

    # print("all_updates shape is ",all_updates.shape)
    # print("n_attacker is ", n_attackers)

    model_re = torch.mean(all_updates, 0)
    model_std = torch.std(all_updates, 0)
    deviation = torch.sign(model_re)
    
    max_vector_low = model_re + 3 * model_std 
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1)
    min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).to(device)

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack([(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.to(device) * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.to(device) * min_rand).T

    print("mal_vec shape is ",mal_vec.shape)

    
    return mal_vec


# def LA_attack_bulyan()