import torch
import numpy as np
import heapq
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
    print("local weights len is ", len(local_weights))
    for update_item in local_weights:
        param_new = []
        for key in std_keys:
            param_new.append(copy.deepcopy(update_item[key]))
            # print("update_item[key] is",update_item[key].size())
        param_update = [] # 清空
        for j in range(len(param_new)):
            sub_res = torch.sub(param_new[j], 0).reshape(-1)
            param_update = sub_res if len(param_update) == 0 else torch.cat((param_update, sub_res), 0)
    # print("param_update size is ",param_update.size())
        param_updates = param_update.clone().unsqueeze(0) if len(param_updates) == 0 else torch.cat((param_updates, param_update.clone().unsqueeze(0)), dim=0)  # 先用unsqueeze(0)增加维度
        # print("param_updates type is ", type(param_updates))
        # print("param_updates shape is ", type(param_updates))
    return param_updates


def restoreWeight(std_dict, std_keys, update_weights):
    # 重构张量，重构字典 
    update_dict = {}
    front_idx = 0
    end_idx = 0
    # mal_update张量重构
    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        tmp_tensor = update_weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = copy.deepcopy(tmp_tensor)
        front_idx = end_idx
    return update_dict



def preKrumGrouping(std_keys, local_weights_delay, local_delay_ew):
    for l in local_delay_ew:
        local_weights_delay.extend(l)
        
    param_updates = modifyWeight(std_keys, local_weights_delay)
    
    return param_updates


'''_keys
对于每个样本，m个clients都会生成对应的的logits，其中有c个恶意clients，
用欧几里得距离为每个client找到m-c-2个与其最近的logits，
最后选择与所有其他clients欧几里得距离之和最小的那个client的logits作为全局的agg_logits；
（对logits整体）
'''
def Krum(para_updates, benign_user_number):  # 
    # clients_l2存储了，某一个client对其他client的l2范式计算
    clients_l2 = [[] for _ in range(len(para_updates))]

    # 求用欧几里得距离为每个client找到m-c-2个与其最近的logits
    for index1, client_logits1 in enumerate(para_updates):
        for index2, client_logits2 in enumerate(para_updates):
            if (index1 == index2):  # 自己和自己不用计算
                continue
            l2 = torch.dist(client_logits1, client_logits2, p=2)  # 计算二范式，就是欧几里德距离
            clients_l2[index1].append(l2)  # clients_l2是list结构

    clients_l2_filter = [[] for _ in range(len(para_updates))]
    for index, client_l2 in enumerate(clients_l2):
        list.sort(client_l2)  # 升序排列，前面的就是最小的
        # print(client_l2)
        client_l2_minN = sum(client_l2[0:benign_user_number - 2])  # 对于单个用户client_l2，对它的前m-c-2个最近的clients求和，作为它与其他client的距离
        # print(client_l2_minN)
        clients_l2_filter[index].append(client_l2_minN)

    # print(clients_l2_filter)
    # 在clients_l2_filter找到最小的
    selected_client_index = clients_l2_filter.index(min(clients_l2_filter))
    print("krum_selected_client_index:", selected_client_index)
    agg_para_update = para_updates[selected_client_index]

    return agg_para_update, selected_client_index



def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix


'''
对于每个样本，m个clients都会生成对应的的logits，其中有c个恶意clients，独立地生成c个恶意的logits。
对于每个样本的logits，剔除m个clients中最大和最小的β个logits（β≥c），
然后计算剩下的m-2β个值的logits均值；
（对样本logits的每一维）
'''
def Trimmed_mean(para_updates, n_attackers):
    sorted_updates = torch.sort(para_updates, 0)[0]
    agg_para_update = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
    return agg_para_update




'''
结合了Krum和Trimmed mean的一个变体。先用Krum找出 x个 ( x 小于等于 m-2c) 用于聚合的本地clients
Krum只是找最小的一个，现在是找最小的x个
然后用Trimmed mean的一个变体来聚合这 x 个本地clients的logits。
聚合的方法是：对于每个样本，先将x个logits中的值进行排序，找到离中位数最近的 y（ y 小于等于alpha-2c）个值，取平均值；
（其实在这个规则里面，就可以看出来，恶意用户的数目c，必须要小于25%；否则这个y肯定小于0）
（对样本logits的每一维）
'''


def Bulyan(temp_all_result, user_number, benign_user_number):  # 在这个规则里面，就可以看出来，恶意用户的数目c，必须要小于25%；否则这个y肯定小于0
    # clients_l2存储了，某一个client对其他client的l2范式计算
    clients_l2 = [[] for _ in range(len(temp_all_result))]

    # 求用欧几里得距离为每个client找到m-c-2个与其最近的logits
    for index1, client_logits1 in enumerate(temp_all_result):
        for index2, client_logits2 in enumerate(temp_all_result):
            if (index1 == index2):
                continue
            l2_distance = torch.dist(client_logits1, client_logits2, p=2)
            clients_l2[index1].append(l2_distance)

    clients_l2_filter = [[] for _ in range(len(temp_all_result))]
    for index, client_l2 in enumerate(clients_l2):
        list.sort(client_l2)  # 升序排列，前面的就是最小的，也就是离他最近的
        client_l2_minN = sum(
            client_l2[0:benign_user_number - 2])  # 对于单个用户client_l2，对它的前m-c-2个最近的clients求和，作为它与其他client的距离
        clients_l2_filter[index].append(client_l2_minN)

    # 在clients_l2_filter找到最小的x个用户,把他们存在selected_clients
    selected_clients = []
    x = 2 * benign_user_number - user_number  # x = m - 2c = m - 2*(m-b) = 2b - m
    for i in range(x):
        selected_client_index = clients_l2_filter.index(min(clients_l2_filter))  # 找到当前的最小值；
        selected_clients.append(temp_all_result[selected_client_index])  # 添加到备选
        clients_l2_filter.pop(selected_client_index)  # 删掉这个数，相当于排除了已经被选择的client

    # 用Trimmed mean的一个变体来聚合这x个本地clients的logits
    y = x - 2 * (user_number - benign_user_number)
    # 对于每个样本，m个clients都会生成对应的的logits;一共有batch个labels，用labels_logits表示
    labels_logits = [[] for _ in range(len(selected_clients[1]))]

    # 为每个label都找到最佳的logits
    for label_index1, label_logits1 in enumerate(selected_clients[0]):
        labels_temp = []
        print("selected_clients[0][0] is", selected_clients[0][0])
        for demission_index2, label_dimensions in enumerate(selected_clients[0][0]):
            labels_dimission_temp = []  # 记录单个label的每一维的平均
            for user_index3, client_logits in enumerate(selected_clients):
                labels_dimission_temp.append(selected_clients[user_index3][label_index1][demission_index2])
            list.sort(labels_dimission_temp)  # 排序一下
            labels_temp.append(sum(labels_dimission_temp[int((x - y) / 2):int(-(x - y) / 2)]) / len(
                labels_dimission_temp[int((x - y) / 2):int(-(x - y) / 2)]))  # 所有client针对某个label的某一维数据求平均
            # 截取找到离中位数最近的 y（ y 小于等于alpha-2c）个值, 相当于截去掉最小的(x-y)/2和最大的(x-y)/2个数, 然后求他们的平均;存到labels_logits中
        labels_logits[label_index1] = labels_temp
    agg_avg_labels = torch.Tensor(labels_logits)  # 把list转成tensor
    return agg_avg_labels


'''
对于每个样本，m个clients都会生成对应的的logits，把这些logits求其中位数，作为全局的agg_logits；
（对样本logits的每一维）
'''


def Median(para_updates): #
    agg_para_update = torch.median(para_updates, 0,keepdim=True)
    print("agg_para is ", agg_para_update[0].squeeze(0))
    return agg_para_update[0].squeeze(0)


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
    print("para updates length is ", len(para_updates))
    avg_para_update = torch.mean(para_updates, dim=0)
    attention_scores = []
    # print("avg size is",avg_para_update.size())
    for index, client_para_update in enumerate(para_updates):
        # print("client size is",client_para_update.size())
        temp_tensor = torch.cosine_similarity(client_para_update, avg_para_update, dim=0).unsqueeze(0)
        # print("temp_tensor is ", temp_tensor)
        attention_scores.append(sum(temp_tensor))  # dim为在哪个维度上计算余弦相似度

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


def pre_AFA(std_keys, current_epoch_updates, current_index, device):
    weight_updates = modifyWeight(std_keys, current_epoch_updates)
    # weight_updates = torch.tensor(weight_updates).to(device)
    print("weight_updates type is ", type(weight_updates))
    AFA_avg, remain_index = AFA(weight_updates, current_index, device)

    return AFA_avg, remain_index
