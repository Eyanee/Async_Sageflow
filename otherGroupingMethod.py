import torch
import numpy as np
from torch import nn
import heapq
import copy
from update import  LocalUpdate
from utils1 import average_weights
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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

def pre_Trimmed_mean(std_dict, std_keys, current_epoch_updates, local_weight_ew):
    for item in local_weight_ew:
        current_epoch_updates.extend(item)

    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Median_avg = Trimmed_mean(weight_updates, 1)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Median_avg, len(weight_updates) - 2

'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求其中位数，作为全局的agg_logits；
（对样本logits的每一维）
'''

def Median(para_updates): #
    agg_para_update = torch.median(para_updates, 0,keepdim=True)
    # print("agg_para is ", agg_para_update[0].squeeze(0))
    return agg_para_update[0].squeeze(0)

def pre_Median(std_dict, std_keys, current_epoch_updates):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    length = len(weight_updates)
    Median_avg = Median(weight_updates)
    Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Median_avg, length

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
    return AFA_avg, len(remain_index)


def preGroupingIndex(local_index_delay, local_index_ew):
    index = local_index_delay
    for idx in local_index_ew:
        index.extend(idx)
    print(index)
    return index

# Zeno
"""


"""
def compute_L2_norm(params):
    squared_sum = 0
    distance = 0

    # 遍历参数字典，计算差异的平方和
    for key in params.keys():
        distance = distance + torch.norm(params[key])

    return distance

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


def test_inference_clone(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    batch_losses = []
    batch_entropy = []
    batch_grad = []

    # with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        # 修改点1：设置模型参数需要梯度
        for param in model.parameters():
            param.requires_grad_(True)


        output, out = model(images)
        # # 构造[batches,categaries]的真实分布向量
        # categaries = output.shape[1]
        Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
        
        entropy  = -1.0 * Information.sum(dim=1) # size [64]
        average_entropy = entropy.mean().item()
        

        batch_loss = criterion(output, labels)
        batch_loss.backward()

        batch_losses.append(batch_loss.item())

        _, pred_labels = torch.max(output,1)
        pred_labels = pred_labels.view(-1)
        pred_dec = torch.eq(pred_labels, labels)
        current_acc = torch.sum(pred_dec).item() + 1e-8

        batch_entropy.append(average_entropy)

        correct += current_acc
        total += len(labels)
        
        # 修改点3：获取并存储梯度
        grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
        # print(grad)
        batch_grad.append(grad)  

        # 修改点3：清零梯度，为下一个batch做准备
        model.zero_grad()

        # 修改点5：恢复模型参数不需要梯度
        for param in model.parameters():
            param.requires_grad_(False)

    xx = batch_grad[0]
    for i in range(1, len(batch_grad)):
        xx += batch_grad[i]
    xx = xx / len(batch_grad)
    return_grad = xx

    accuracy  = correct/total

        

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy), return_grad

def Zeno(weights, grad, args, model, cmm_dataset, current_epoch):

    common_acc, common_loss, _, gd = test_inference_clone(args, model, cmm_dataset)
    
    test_model = copy.deepcopy(model)
    loss_list = []
    for param in weights:
        mod_params = copy.deepcopy(model.state_dict())
        for key in param.keys():
            mod_params[key] = torch.subtract(mod_params[key],param[key]*0.01)
        test_model.load_state_dict(mod_params)
        test_acc,test_loss, _ ,gd = test_inference_clone(args, model, cmm_dataset)
        loss_list.append(test_loss)
        
    print("loss_list is", loss_list)
    print("common_loss is", common_loss)
    
    fai = 0.01
    w = model.state_dict()
    score = []

    for i in range(0, len(weights)):
        length = compute_L2_norm(weights[i])
        print("length is", length)
        # length = compute_gradient(w, weights[i], args.lr)
        tmp = common_loss - loss_list[i] - fai * length
        print("score is ",tmp)
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

    alpha = 0.5
    if args.dataset == 'cifar10' or 'cifar100':
        alpha = 0.5 ** (current_epoch // 300)
    elif args.dataset =='fmnist':
        alpha = 0.5 ** (current_epoch // 20)
    elif args.dataset =='mnist':
        alpha = 0.5 ** (current_epoch // 15)

    for key in w.keys():       
        re_model[key] = re_model[key] * (alpha) + w[key] * (1 - alpha)
    
    return re_model

def pre_Zeno(current_epoch_updates, args, loss, cmm_dataset, global_model):
    # weight_updates = modifyWeight(std_keys, current_epoch_updates)
    Zeno_avg = Zeno(current_epoch_updates, loss, args, global_model, cmm_dataset)
    # Median_avg = restoreWeight(std_dict, std_keys, Median_avg)
    return Zeno_avg


# Zeno++
"""


"""
def Zenoplusplus(args, global_state_dict, grad_updates,indexes):
    print("index is ", indexes)
    # parameters for zeno++
    zeno_rho = 0.001 #
    zeno_epsilon = 0 #

    accept_list = []
    # scaling the update:
    global_param_square = 0
    user_param_square = 0
    for param_update in grad_updates:
        for key in global_state_dict.keys():
            global_param_square = global_param_square + torch.sum(torch.square(global_state_dict[key]))
            user_param_square = user_param_square + torch.sum(torch.square(param_update[key]))

        c = torch.sqrt(user_param_square / global_param_square)
        print("c is ", c)
        for key in param_update.keys():
                param_update[key] = param_update[key] * c
        
        # compute score
        zeno_innerprod = 0
        zeno_square = global_param_square
        # 计算内积
        for key in global_state_dict.keys():
            print("调试 - global_state_dict[key] 形状:", global_state_dict[key].shape)
            print("调试 - param_update[key] 形状:", param_update[key].shape)
            global_state_dict[key] = torch.transpose(global_state_dict[key])
            zeno_innerprod = zeno_innerprod + torch.tensordot(global_state_dict[key], param_update[key])
        #计算分数
        score = args.lr * (zeno_innerprod) - zeno_rho * (zeno_square) + args.lr * zeno_epsilon
        print("score :", score)
        # 分数大于0则接收
        if score >= 0:
            print("accept")
            accept_list.append(param_update)
    # 怎么修改？
    return accept_list



def compute_mmd(x, y, sigma=1.0):
    # 计算高斯核矩阵
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    xy = torch.matmul(x, y.t())

    # 计算高斯核函数
    k_xx = torch.exp(-torch.sum((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-torch.sum((y.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2) / (2 * sigma ** 2)).mean()

    # 计算MMD
    mmd = k_xx + k_yy - 2 * k_xy
    return mmd


# FLARE
"""


"""
def FLARE(args, global_model, param_updates, common_dataset):
    ### 使用客户端模型计算辅助数据集上的每张图像的PLR，组成一个矩阵

    
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    test_model = copy.deepcopy(global_model)
    test_dict = copy.deepcopy(global_model.state_dict())
    user_len = len(param_updates)
    test_model.eval()

    trainloader = DataLoader(common_dataset, batch_size=64, shuffle=False) #待修改
    user_PLR = []

    for param in param_updates:
        test_model.load_state_dict(param)
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            output,out, PLR = test_model(images)
            # print("round ", batch_idx)
            # print("PLR size ", PLR.shape)
        user_PLR.append(PLR)
    print("user_PLR len size", len(user_PLR))

    mmd_set = np.zeros((user_len, user_len))
    mmd_indicator = np.zeros((user_len, user_len))
    count_indicator = np.zeros(user_len)
    
    ## MMD 矩阵赋值
    for i in range(user_len):
        for j in range(i+1, user_len):
            mmd_value = compute_mmd(user_PLR[i], user_PLR[j])
            print("mmd_value", mmd_value)
            mmd_set[i,j]= mmd_value
            mmd_set[j,i]= mmd_value
    
    ## 
    k = int(user_len * 0.5)
    for idx, row in  enumerate(mmd_set):
        # 对张量进行降序排序  
        sorted_row = np.sort(row)  
        kth_largest = sorted_row[k - 1]    
        for jdx, element in enumerate(row):
            if element >= kth_largest:
                mmd_indicator[idx, jdx] = 1
                count_indicator[jdx] = count_indicator[jdx] + 1 ## 被作为最近邻
    ## 加权聚合
    count_tensor = torch.Tensor(count_indicator)
    count_res = F.softmax(count_tensor, dim=-1)
    # print("count_res", count_indicator)
    print("count_res", count_res)
    # print("count_sum is ", torch.sum(count_res))
    for key in test_dict.keys():
        for i in range(0, len(count_res)):
            if i == 0:
                test_dict[key] = count_res[i] * param_updates[i][key]
            else:
                test_dict[key] += count_res[i] * param_updates[i][key]


        
    return test_dict

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
    # 返回缩放后的模型参数字典+

    
    return scaled_model

def FLTrust(weights, grad, args, model, train_dataset, dict_common, epoch, indexes):
    # 本质上吧cs的离群值全部剔除了

    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_common, idx=0, data_poison=False)
    w_global = model.state_dict()
    acc , loss, _, gd = test_inference_clone(args, model, DatasetSplit_clone(train_dataset,dict_common) )
    TS = []
    
    for i in range(0, len(grad)):
        tt = torch.cosine_similarity(gd, grad[i], dim=0, eps=1e-8)
        print("grad is {}, index {}".format(tt, indexes[i]))
        TS.append(tt)

        # weights[i] = normalize_update(w_global, weights[i], w, grad[i], gd, args)
        
        change = torch.norm(grad[i])
        bound = torch.norm(gd)
        if bound < change :
            for key in weights[i].keys():    
                weights[i][key] = w_global[key] * (1 - bound/change) + weights[i][key] * (bound/change)

    relu = nn.ReLU(inplace=True)
    TS = torch.Tensor(TS)
    TS = relu(TS)

    re_model = copy.deepcopy(w_global)
    for key in w_global.keys():
        for i in range(0, len(TS)):
            if i == 0:
                re_model[key] = TS[i] * weights[i][key]
            else:
                re_model[key] += TS[i] * weights[i][key]

        re_model[key] = re_model[key] / sum(TS)

    alpha = 0.5
    if args.dataset == 'cifar10' or 'cifar100':
        alpha = 0.5 ** (epoch // 300)
    elif args.dataset =='fmnist':
        alpha = 0.5 ** (epoch // 20)
    elif args.dataset =='mnist':
        alpha = 0.5 ** (epoch // 15)

    for key in w_global.keys():
        re_model[key] = re_model[key] * (alpha) + w_global[key] * (1 - alpha)

    return re_model

def update_weights(global_model, param_update):
    alpha = 0.5
    return_params = copy.deepcopy(global_model.state_dict())
    for key in param_update.keys():
        return_params[key] =  return_params[key] * (1 - alpha) + param_update[key] * (alpha)

    return return_params

def AFLGuard(grad_param_updates, global_model, global_test_model, epoch, lamda):
    global_weights = copy.deepcopy(global_model.state_dict())
    for gd_param in grad_param_updates:
        w, loss, gd_server = global_test_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )
        # print("len(gd param)", len(gd_param[1]))
        norm_1 = torch.norm(torch.subtract(gd_server , gd_param[1]))
        norm_2 = torch.norm(gd_server)

        if norm_1 <= norm_2 * lamda:
            # 满足条件则更新全局模型
            # print("param type ", gd_param[0].shape)
            global_weights = update_weights(global_model,gd_param[0])
            global_model.load_state_dict(global_weights)

    return global_weights


# norm-clipping/ norm-clipping
# 根据一定Bound对参数进行剔除
def norm_clipping(global_model, local_weights_delay ,local_delay_ew):
    
    params = copy.deepcopy(local_weights_delay)
    for item in local_delay_ew:
        params.extend(item)

    number_to_consider = int(len(params)* 0.8) ### 尝试0.5
    std_keys = get_key_list(global_model.state_dict().keys())
    weight_updates = modifyWeight(std_keys, params)
    print("weight size is ", weight_updates.shape)
    norm_res = torch.norm(weight_updates, p =2 ,dim = 1)
    print("norm res size is ", norm_res.shape)
    sorted_norm, sorted_idx = torch.sort(norm_res)
    used_idx = sorted_idx[:number_to_consider]
    print("used idx is ", used_idx)
    avg_weight =  torch.mean(weight_updates[used_idx, :],dim = 0)
    print("weight size is ", avg_weight.shape)
    weight_res = restoreWeight(copy.deepcopy(global_model.state_dict()),std_keys,avg_weight)

    return weight_res

def flatten_parameters(model):
    return np.concatenate([param.cpu().detach().numpy().flatten() for param in model.parameters()])

def is_nested_list(lst):  

    for item in lst:  

        if isinstance(item, list):  

            return True  

    return False 

class DatasetSplit_clone(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

