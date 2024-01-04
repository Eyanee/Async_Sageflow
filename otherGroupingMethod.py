import torch
import numpy as np
import heapq
import copy
from update import test_inference, LocalUpdate
from utils1 import average_weights


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



def preGrouping(std_keys, local_weights_delay, local_delay_ew):
    for l in local_delay_ew:
        local_weights_delay.extend(l)
        
    param_updates = modifyWeight(std_keys, local_weights_delay)
    
    return param_updates





'''
对于每个样本，m个clients都会生成对应的的logits，其中有c个恶意clients，独立地生成c个恶意的logits。
对于每个样本的logits，剔除m个clients中最大和最小的β个logits（β≥c），
然后计算剩下的m-2β个值的logits均值；
（对样本logits的每一维）
'''
def Trimmed_mean(para_updates, n_attackers = 1):
    sorted_updates = torch.sort(para_updates, 0)[0]
    agg_para_update = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
    return agg_para_update

def pre_Trimmed_mean(std_dict, std_keys, current_epoch_updates):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Median_avg = Trimmed_mean(, 1)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Median_avg, len(weight_updates) - 2

'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求其中位数，作为全局的agg_logits；
（对样本logits的每一维）
'''

def Median(para_updates): #
    agg_para_update = torch.median(para_updates, 0,keepdim=True)
    print("agg_para is ", agg_para_update[0].squeeze(0))
    return agg_para_update[0].squeeze(0)

def pre_Median(std_dict, std_keys, current_epoch_updates):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Median_avg = Median(weight_updates)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Median_avg 

'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求平均，作为全局的agg_logits；
（对样本logits的每一维）
'''
def Mean(para_updates, args):  #
    agg_para_update = torch.mean(para_updates, dim=0)
    return agg_para_update



'''
Adaptive federated average（AFA）：对所有的logits求平均，
然后计算每一个clients的logits与平均logits的余弦相似度，
剔除离群值（剔除多少？百分比？20%？），
剩下的再求平均，作为全局的agg_logits；（对logits整体）
'''


def AFA(para_updates, interfere_idx, device):  #

    # 两个向量有相同的指向时，余弦相似度的值为1；
    # 两个向量夹角为90°时，余弦相似度的值为0；
    # 两个向量指向完全相反的方向时，余弦相似度的值为-1。
    # 这结果是与向量的长度无关的，仅仅与向量的指向方向相关。
    # 余弦相似度通常用于正空间，因此给出的值为-1到1之间。
    avg_para_update = torch.mean(para_updates, dim=0)
    attention_scores = []
    # print("avg size is",avg_para_update.size())
    for index, client_para_update in enumerate(para_updates):
        # print("client size is",client_para_update.size())
        temp_tensor = torch.cosine_similarity(client_para_update, avg_para_update, dim=0).unsqueeze(0)
        # print("temp_tensor is ", temp_tensor)
        attention_scores.append(sum(temp_tensor))  # dim为在哪个维度上计算余弦相似度

    # print("attention score value is ", attention_scores)
    # 剔除离群值
    # 先求要删除多少个数，记为 abandon_count
    abandon_count = int(len(attention_scores) * 0.2)  # 目前是剔除百分之20
    # 然后，找到attention_score中倒数第n个数——在这里是前20%的分界点；不能用排序，attention_scores的顺序不要变
    arr_min_list = heapq.nsmallest(abandon_count, attention_scores)  ##获取最小的abandon_count个值并按升序排序
    abandon_num_flag = arr_min_list[-1]  # arr_min的最后一个值就是分界点
    # 记录大于分界点的数对应的下标
    filter_index = []
    for index, attention_score in enumerate(attention_scores):
        if (attention_score > abandon_num_flag):
            filter_index.append(index)
    # 用这个下标去寻找对应的client_logits,放到filter_clients中
    filter_out = list()
    filter_left = list()
    for idx in range(len(para_updates)):
        if idx not in filter_index:
            filter_out.append(interfere_idx[idx])
        else:
            filter_left.append(interfere_idx[idx]) # 剩下的原有idx
    
    filter_clients = []
    count = 0
    # print("benign number is ", benign_user_number)
    for index in filter_index:
        filter_clients.append(para_updates[index])

    agg_para_update = torch.mean(torch.tensor([item.cpu().detach().numpy() for item in filter_clients]).to(device), dim=0)  # 。原因是：要转换的list里面的元素包含多维的tensor。
    return agg_para_update, filter_left


def pre_AFA(std_dict, std_keys, current_epoch_updates, current_index, device):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    AFA_avg, remain_index = AFA(weight_updates, current_index, device)
    AFA_avg = restoreWeight(std_dict, std_keys, AFA_avg)
    return AFA_avg, remain_index


def preGroupingIndex(local_index_delay, local_index_ew):
    index = local_index_delay
    for idx in local_index_ew:
        index.extend(idx)
    print(index)
    return index

# Zeno
"""


"""
def compute_gradient(model1, model2, lr):
    # 用于存储梯度模长
    len = 0
    # 遍历模型参数字典的键
    for key in model1.keys():
        # 取出对应的参数张量
        param1 = model1[key]
        param2 = model2[key]
        # 根据公式计算梯度
        grad = (param1 - param2) / lr
        len += torch.norm(grad)

    # 梯度模长
    return len


def Zeno(weights, loss, args, model, cmm_dataset):

    common_acc, common_loss, _ = test_inference(args, model, cmm_dataset)
    fai = 0.0
    w = model.state_dict()
    score = []

    for i in range(0, len(weights)):
        length = compute_gradient(w, weights[i], args.lr)
        tmp = common_loss - loss[i] - fai * length
        score.append(tmp.__float__())

    # print("# test1", type(score[0]))
    # 找到张量的最小值, 将score全部变为正数
    min_value = min(score)
    score = np.array(score) - min_value
    score = score / sum(score)

    re_model = copy.deepcopy(w)
    for key in w.keys():
        for i in range(0, len(score)):
            if i == 0:
                re_model[key] = score[i] * weights[i][key]
            else:
                re_model[key] += score[i] * weights[i][key]

    return re_model

def pre_Zeno(current_epoch_updates, args, loss, cmm_dataset, global_model):
    # weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Zeno_avg = Zeno(current_epoch_updates, loss, args, global_model, cmm_dataset)
    # Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Zeno_avg


# Zeno++
"""


"""
def Zenoplusplus(args, global_state_dict, grad_updates):
    # parameters for zeno++
    zeno_rho = 0.001 #
    zeno_epsilon = 0 #

    accept_list = []
    # scaling the update:
    param_square = 0
    zeno_param_square = 0
    for param_update in grad_updates:
        for param_g, param_u in zip(global_state_dict, param_update):
            if param.grad_req != 'null':
                global_param_square = param_square + torch.sum(torch.square(param_g))
                user_param_square = zeno_param_square + torch.sum(torch.square(param_u))
        c = torch.sqrt(user_param_square.asscalar() / global_param_square.asscalar())
        for param in param_update:
            if param.grad_req != 'null':
                grad = param.grad()
                grad[:] = grad * c
        
        # compute score
        zeno_innerprod = 0
        zeno_square = param_square
        # 计算内积
        for param, zeno_param in zip(global_state_dict, param_update):
            if param.grad_req != 'null':
                zeno_innerprod = zeno_innerprod + torch.matmul(param.grad() * zeno_param.grad())
        #计算分数
        score = args.lr * (zeno_innerprod.asscalar()) - zeno_rho * (zeno_square.asscalar()) + args.lr * zeno_epsilon
        # 分数大于0则接收
        if score >= 0:
            print("accept")
            accept_list.append(param_update)
    # 怎么修改？
    return accept_list





# FLTrust
"""

"""
def cosine_similarity(model1, model2):

    # 初始化一个空列表，用于存储每个参数的余弦相似度
    cos_sim_list = []
    # 遍历模型参数字典的键
    for key in model1.keys():
        # 取出对应的参数张量
        param1 = model1[key]
        param2 = model2[key]

        # 利用torch内置的cosine_similarity函数，计算两个张量的余弦相似度
        # dim=0表示按列计算，eps=1e-8表示添加一个小的正数，防止除零错误
        cos_sim = torch.cosine_similarity(param1, param2, dim=0, eps=1e-8)
        # print("#test", cos_sim)
        # 将计算结果添加到列表中
        cos_sim_list.append(torch.mean(cos_sim))
    # 返回余弦相似度列表
    return cos_sim_list


def normalize_update(update1, update2):

    # 初始化一个空字典，用于存储缩放后的模型参数
    scaled_model = {}
    # 遍历模型参数字典的键
    for key in update1.keys():
        # 取出对应的参数张量
        param1 = update1[key]
        param2 = update2[key]
        # 计算两个更新的L2范数
        norm1 = torch.norm(param1)
        norm2 = torch.norm(param2)
        # 如果范数不相等，需要进行缩放
        if norm1 != norm2:
            # 计算缩放因子，即范数之比
            scale_factor = norm2 / norm1
            # 使用torch的mul函数，对update1进行缩放，使其与update2的范数相同
            scaled_param = torch.mul(param1, scale_factor)
        else:
            # 如果范数相等，不需要缩放，直接复制update1
            scaled_param = param1.clone()
        # 将缩放后的张量存入scaled_model字典中
        scaled_model[key] = scaled_param
    # 返回缩放后的模型参数字典
    return scaled_model

# weights是一个list, 每个元素为一组权重
def FLTrust(weights, args, model, cmm_dataset, dict_common, epoch):

    local_model = LocalUpdate(args=args, dataset=cmm_dataset, idxs=dict_common, idx=0, data_poison=False)
    w, loss = local_model.update_weights(model=copy.deepcopy(model), global_round=epoch)
    TS = []
    for i in range(0, len(weights)):
        tt = cosine_similarity(weights[i], w)
        TS.append(sum(tt)/len(tt))
        weights[i] = normalize_update(weights[i], w)

    relu = torch.nn.ReLU(inplace=True)
    TS = torch.Tensor(TS)
    TS = relu(TS)

    re_model = copy.deepcopy(w)
    for key in w.keys():
        for i in range(0, len(TS)):
            if i == 0:
                re_model[key] = TS[i] * weights[i][key]
            else:
                re_model[key] += TS[i] * weights[i][key]

        re_model[key] = re_model[key] / sum(TS)

    return re_model


# norm-clipping/ norm-clipping
# 根据一定的阈值对于提交的参数进行裁剪
# def mean_norm(net, nfake, sf):

   