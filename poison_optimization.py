import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy
import math
from update import test_inference
from otherGroupingMethod import get_key_list
from p_Optimizer import MyPOptimizer



def cal_similarity(ori_params, mod_params):
    std_keys = get_key_list(ori_params.keys())
    params1 = torch.cat([ori_params[k].view(-1) for k in std_keys])
    params2 = torch.cat([mod_params[k].view(-1) for k in std_keys])

    # 计算余弦相似度
    cos_similarity = F.cosine_similarity(params1, params2, dim=0)

    return cos_similarity

def model_dist_norm(ori_params, mod_params):
    squared_sum = 0
    distance = 0

    pdlist = nn.PairwiseDistance(p=2)
    # 遍历参数字典，计算差异的平方和
    for key in ori_params.keys():
        if ori_params[key].ndimension() == 1:
            t1 = ori_params[key].unsqueeze(0)
            t2 = mod_params[key].unsqueeze(0)

        else:
            t1 = ori_params[key]
            t2 = mod_params[key]
        temp1 = torch.sum(pdlist(t1 ,t2))
        output = torch.sum(temp1)
        distance += output

    return distance


def get_distance_list(ori_state_dict, mod_state_dict):
    squared_sum = 0
    distance = 0
    distance_list = []
    pdlist = nn.PairwiseDistance(p=2)
    # 遍历参数字典，计算差异的平方和
    for key in ori_state_dict.keys():

        if ori_state_dict[key].ndimension() == 1:
            t1 = ori_state_dict[key].unsqueeze(0)
            t2 = mod_state_dict[key].unsqueeze(0)
        else:
            t1 = ori_state_dict[key]
            t2 = mod_state_dict[key]
        temp1 = torch.sum(pdlist(t1 ,t2))
        output = torch.sum(temp1)
        distance_list.append(output)

    return distance_list


"""
流程框架函数
"""
def Outline_Poisoning(args, pre_global_model, global_model, malicious_models, train_dataset, distance_ratio, pinned_accuracy_threshold, adaptive_accuracy_threshold, last_round_poison_w, poisoned):

    ref_model = global_model # 已经在调用阶段deepcopy过的  恶意用户本地训练结果的加权结果（本轮）
    pre_distance_ratio = distance_ratio
    # indicator_res, new_distance_ratio = test_attack_result(pre_global_model, ref_model, distance_ratio, poisoned)
    new_distance_ratio = distance_ratio
    # if indicator_res:
    #     print("using last round poisoned parameters")
    #     global_model.load_state_dict(last_round_poison_w)
    #     test_acc, test_loss, test_entropy = test_inference(args, global_model, train_dataset)
    #     print("test_acc is ",test_acc)
    #     return last_round_poison_w, new_distance_ratio
    distance_threshold = cal_ref_distance(malicious_models, ref_model, new_distance_ratio) # 计算参考L2 distance 门槛 

    print("calculated distance threshold is ", distance_threshold)
    # print("pinned_accuracy threshold is ", pinned_accuracy_threshold)

    w_rand = add_small_perturbation(global_model, args, pinned_accuracy_threshold, train_dataset, perturbation_range=(-0.1, 0.1))
    #使用accuracy_list 更新adaptive accuracy threshold?  暂时先不用accuracy list

    w_poison, optimization_res = phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold, adaptive_accuracy_threshold, pinned_accuracy_threshold)
    # 如果没有成功优化，则下一轮的distance不应该被改变
    if not optimization_res:
        # distance ratio 不应该被改变
        new_distance_ratio = pre_distance_ratio
        # 重新生成 参考恶意模型
        # adaptive_accuracy_threshold
    # else accuracy 试着降低
    else:
        return w_poison, new_distance_ratio

    print("new distance ratio is", new_distance_ratio)
    return w_poison, new_distance_ratio

def Outline_Poisoning_compare(args, pre_global_model, global_model, malicious_models, train_dataset, distance_ratio, pinned_accuracy_threshold, adaptive_accuracy_threshold, poisoned):
    w_rand = add_small_perturbation(global_model, args, pinned_accuracy_threshold, train_dataset, perturbation_range=(-0.1, 0.1))

    return w_rand, distance_ratio
"""
用于检测上轮投毒是否生效的函数
distance增加幅度可能需要调整
可以使用 alpha和beta两次值进行决定
马尔科夫决策过程
"""
def test_attack_result(pre_global_model, global_model, distance_ratio, poisoned):
    print("distance ratio is" , type(distance_ratio))
    if poisoned :
        attack_res, _ = Indicator(pre_global_model, global_model)
        print("attack res is" , attack_res)
        if attack_res:
            distance_ratio = distance_ratio * 1.1

        else :
            distance_ratio = distance_ratio / 1.1
        return attack_res, distance_ratio
    else:
        return False, distance_ratio

def Indicator(fi_global_model, global_model):
    """
    通过 FGNV 来判断本轮投毒是否生效
    
    返回 投毒是否生效 True/ False
    
    需要确定门槛值 δ 需要重新确定
    
    """
    case = 1

    if case == 1:
        delta = 0.002
        FGNV_pre = cal_FGNV(fi_global_model.state_dict())
        FGNV_cur = cal_FGNV(global_model.state_dict())
    
        compare_res = math.fabs((FGNV_cur - FGNV_pre)/FGNV_pre) # 取绝对值
        print("compare_res is ", compare_res)

        if compare_res > delta:
            return True, compare_res
        else:
            return False, compare_res
    elif case == 2:
        delta = 0.005
        FGNV_sum = 0
        std_keys = get_key_list(global_model.state_dict())
        fi_dict = fi_global_model.state_dict()
        cur_dict = global_model.state_dict()
        for key in std_keys:
            FGNV_sum += torch.sum(torch.abs(torch.sub(fi_dict[key], cur_dict[key])))
            

        FGNV_base = cal_FGNV(fi_dict)
        compare_res = math.fabs(FGNV_sum / FGNV_base) # 改变量的比例
        if FGNV_sum > delta:
            return True, compare_res
        else:
            return False, compare_res
    
    elif case == 3:
        delta = 0.005
        FGNV_sum = 0
        std_keys = get_key_list(global_model.state_dict())
        fi_dict = fi_global_model.state_dict()
        cur_dict = global_model.state_dict()
        for key in std_keys:
            FGNV_sum += torch.sum(torch.pow(torch.sub(fi_dict[key], cur_dict[key]), 2))


        FGNV_base = cal_FGNV(fi_dict)
        compare_res = math.fabs(FGNV_sum / FGNV_base) # 改变量的比例
        if FGNV_sum > delta:
            return True, compare_res
        else:
            return False, compare_res

def cal_FGNV(model_dict):
    res = 0
    for key in model_dict.keys():
        res += torch.norm(model_dict[key])
    return res

def cal_ref_distance(malicious_models, global_model, distance_ratio):
    
    distance_res = computeTargetDistance(malicious_models, global_model, distance_ratio)

    

    return distance_res


"""
自蒸馏 + adpative clipping
得到符合要求的投毒结果
"""
def phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold, adaptive_accuracy_threshold, pinned_accuracy_threshold):

    # parameter determination
    round = 0
    MAX_ROUND = 10
    entropy_threshold = 2.0
     # 准备教师模型
    teacher_model = copy.deepcopy(global_model)
    # 准备学生模型 
    student_model = copy.deepcopy(global_model)

    teacher_model.load_state_dict(w_rand)
    student_model.load_state_dict(w_rand)
    teacher_model.eval()

    for round in range(MAX_ROUND):

        test_acc, test_loss, test_entropy = test_inference(args, student_model, train_dataset)
        test_distance = model_dist_norm(student_model.state_dict(), global_model.state_dict())
        test_simliarity = cal_similarity(student_model.state_dict(), global_model.state_dict())
        print("________________________________")
        print("test distance is ", test_distance)
        print("test similarity is ", test_simliarity)
        print("test acc is ", test_acc)
        print("test loss is", test_loss)
        print("test entropy is ", test_entropy)
        print("________________________________")
        if test_distance <= distance_threshold and test_acc <= pinned_accuracy_threshold and test_entropy  <= entropy_threshold:
            return student_model.state_dict(), True
        elif test_entropy > entropy_threshold:
            distillation_res, w_rand = self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distillation_round = 10)
        elif test_distance > distance_threshold:
            w_rand = adaptive_scaling(w_rand, global_model.state_dict(), distance_threshold, test_distance)
            student_model.load_state_dict(w_rand)
        else:
            distillation_res, w_rand = self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distillation_round = 10)
            # 只有accuracy 不满足条件
            # 暂时不处理
        student_model.load_state_dict(w_rand)
    # 返回值为模型参数
    print("reach max round") # accuracy在这里调整
    return student_model.state_dict(), False


"""
接入两种方案进行尝试
"""
def adaptive_scaling(w_rand, ref_model_dict, distance_threshold, test_distance):
    use_case = 5
    """
    case 1 按照distance进行排序
    优先保留差距较大的向量值
    """
    pdlist= nn.PairwiseDistance(p=2)
    if use_case == 1:
        distance_list = []
        sum = 0
        keys = get_key_list(ref_model_dict)
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2))
            print("diff is ",diff)
            distance_list.append(diff)
        
        sorted_list = sorted(distance_list)
        distance_diff = test_distance - distance_threshold
        threshold = distance_diff #
        print("threshold is ",threshold)
        for i in range(len(distance_list)):
            sum += sorted_list[i]
            print("sum is", sum)
            if sum >= distance_diff:
                threshold = sorted_list[i]
                ratio = (sum - distance_diff) / sorted_list[i]
                print("ratio is", ratio)
                print("threshold is", threshold)
                break
        if sum < distance_diff:
            print("fail to generate ")
            
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2).float())
            print("cal diff is", diff)
            if diff < threshold: #直接舍弃
                print("discard")
                w_rand[key] = ref_model_dict[key]
            elif diff >= threshold:
                print("clipping")
                w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
        return w_rand
    
    # case 2 优先裁剪 差异较大的向量层          
    elif use_case == 2:
        distance_list = []
        pdlist= nn.PairwiseDistance(p=2)
        distance_diff = test_distance - distance_threshold
        sum_d = 0
        sum_s = 0
        keys = get_key_list(ref_model_dict)
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2))
            print("diff is ",diff)
            distance_list.append(diff)
            sum_s += diff
        print("sum_s is", sum_s)
        sorted_list = sorted(distance_list, reverse = False)  # 降序排列
        for item in sorted_list:
            sum_d = sum_d + item
            if sum_d >= distance_diff:
                sum_d = sum_d - item
                break
            threshold = item
            print("item is", item)
        ratio = math.sqrt((distance_threshold - sum_d) /(sum_s - sum_d).double())
  
        if sum_d < distance_diff:
            print("fail to generate")
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2).float())
            print("cal diff is", diff)
            if diff >= threshold:
                print("clipping")
                w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
        return w_rand
    
    ## use case 3: 优先裁剪靠前的层
    elif use_case == 3:
        pdlist= nn.PairwiseDistance(p=2)
        distance_diff = test_distance - distance_threshold
        sum_d = 0
        keys = get_key_list(ref_model_dict)
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2))
            sum_d += diff
            if sum_d >= distance_threshold:
                ratio = distance_threshold/sum_d
                target_key = key
                break
        if sum_d < distance_threshold:
            print("fail to generate")
        for key in keys:
            w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
            if key == target_key:
                break
        return w_rand

    ## use case 4: 优先裁剪靠后的层
    elif use_case == 4:
        pdlist= nn.PairwiseDistance(p=2)
        distance_diff = test_distance - distance_threshold
        sum_d = 0
        keys = reversed(get_key_list(ref_model_dict))
        for key in keys:
            t1 = ref_model_dict[key]
            t2 = w_rand[key]
            if w_rand[key].ndimension() == 1:
                t1 = t1.unsqueeze(0)
                t2 = t2.unsqueeze(0)
            diff = torch.sum(pdlist(t1, t2))
            sum_d += diff
            if sum_d >= distance_threshold:
                ratio = distance_threshold/sum_d
                target_key = key
                break
        if sum_d < distance_threshold:
            print("fail to generate")
        for key in keys:
            w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
            if key == target_key:
                break
        return w_rand
    elif use_case == 5:
        pdlist= nn.PairwiseDistance(p=2)
        cal_distance = test_distance
        while cal_distance > distance_threshold:
            ratio = math.sqrt((distance_threshold / test_distance).double())
            keys = reversed(get_key_list(ref_model_dict))
            for key in keys:
                w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
            cal_distance = model_dist_norm(w_rand, ref_model_dict)
            print("cal_distance is ", cal_distance)
        return w_rand
    return w_rand


def self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, accuracy_threshold, distillation_round):
    """
    自蒸馏函数主体
    """
    ### 自蒸馏参数
    lr = args.lr
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    # optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = MyPOptimizer(student_model.parameters(),lr=lr)
    criterion1 = nn.NLLLoss().to(device)
    # criterion2 = nn.functional.kl_div().to(device)
    teacher_model.to(device)
    student_model.to(device)
    
    num_epochs = distillation_round
    temperature = 50 ### 增大
    alpha = 0.01
    beta = 0.5


    student_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        acc, loss, avg_entropy = test_inference(args, student_model, train_dataset)
        compute_distance = model_dist_norm(student_model.state_dict(), ref_model.state_dict())
        print("++++++++++++++++++++")
        print("after training ",epoch)
        print("avg test entropy is ", avg_entropy)
        print("avg test loss is ", loss)
        print("avg accuracy is ", acc) 
        print("compute_distance is ",compute_distance)
        print("++++++++++++++++++++")

        if avg_entropy <= entropy_threshold and acc <= accuracy_threshold:
           
            return True, student_model.state_dict()

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_model.zero_grad()
            teacher_labels = list()
            _ , teacher_outputs = teacher_model(images)
            for item in teacher_outputs:
                pred_label = int(torch.max(item, 0)[1])
                teacher_labels.append(pred_label)

            pred_is = torch.tensor(teacher_labels)
            pred_is = pred_is.to(device)
            stu_out, student_outputs = student_model(images)
            ref_out, ref_outputs = ref_model(images)
            _ , teacher_outputs = teacher_model(images)
            # loss = alpha * criterion1(stu_out, pred_is)  + beta * temperature** 2 * soft_loss(student_outputs, teacher_outputs, temperature = 5)
            # d_loss = model_dist_norm(ref_model.state_dict(), student_model.state_dict())
            # print("d_loss is", d_loss)
            l_loss = criterion1(stu_out, pred_is)

            loss = l_loss
            
            loss.backward()
            optimizer.step(list(ref_model.parameters()))

    # 设置False出口
    return True, student_model.state_dict()


def soft_loss(student_outputs, teacher_outputs, temperature):
    """
    自蒸馏的损失函数
    使用hard targets 和 soft targets相结合
    L = α * L_soft + β * L_hard

    Args:
        student_outputs (torch.Tensor): 学生模型的输出
        teacher_outputs (torch.Tensor): 教师模型的软标签
        temperature (float): 温度参数，用于调节软标签的相对尺度

    Returns:
        torch.Tensor: 自蒸馏损失
    """
    # 计算学生模型和教师模型的软标签的 softmax
    stu_s = F.log_softmax(student_outputs/temperature, dim = 1)

    teacher_s = F.softmax(teacher_outputs/temperature, dim = 1)

    # loss 采用的是KL散度来计算差异
    loss = F.kl_div(stu_s, teacher_s, size_average=False) * (temperature) / student_outputs.shape[0]
    
    return loss
    

def add_small_perturbation(original_model, args, target_accuracy, train_dataset, perturbation_range=(-0.1, 0.1)):
    """
    在原有张量上添加较小的扰动，使得新生成的张量在
    1. 欧氏距离
    与原向量保证一定的相似度
    """
    # std_keys = original_model.state_dict().keys()

    orignal_state_dict = original_model.state_dict()
    test_model = copy.deepcopy(original_model)
    perturbed_dict = copy.deepcopy(orignal_state_dict)
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'

    max_iterations = 100
    MAX_ROUND = 3
    MAX_SUCC_ROUND = 5
    succ_round = 0
    iteration = 0
    min_acc = 1
    # reverse_keys = reversed(list(orignal_state_dict.keys())) 
    reverse_keys = list(orignal_state_dict.keys())

    while iteration < max_iterations:
        # 计算当前扰动范围的中点
        mid_min = perturbation_range[0] * 2
        print("mid_min is", mid_min)
        mid_max = mid_min * -1

        # 生成中点扰动张量
        for round in range(MAX_ROUND):
            for idx, key in enumerate(reverse_keys):
                if idx == 2 or idx == 3:
                    temp_original = orignal_state_dict[key]
                    perturbs = torch.tensor(np.random.uniform(low=mid_min, high=mid_max, size=temp_original.shape)).to(device)
                    # perturbs = torch.randn_like(temp_original)* (mid_max - mid_min) + (mid_min + mid_max) / 2
                    # perturbs = perturbs.to(device)
                    perturbs = torch.add(perturbs, temp_original)
                    perturbed_dict[key] = perturbs

            # 计算新张量和原张量之间的相似性
            test_model.load_state_dict(perturbed_dict)
            acc, loss, entropy = test_inference(args, test_model, train_dataset)
            print("====================")
            print("iteration is ", iteration)
            print("round is ",round )
            print("accuracy is ", acc)
            print("====================")

            # 判断是否达到目标相似性
            if acc <= target_accuracy:
                succ_round = succ_round + 1
                if acc < min_acc:
                    print("succ_min + 1")
                    min_acc = acc
                    final_dict = copy.deepcopy(perturbed_dict)
                if succ_round >= MAX_SUCC_ROUND:
                    return final_dict

            else:
                # 相似性不满足要求，更新扰动范围，继续迭代
                perturbation_range = (mid_min, mid_max)

        iteration = iteration + 1

        # 若迭代次数超过最大迭代次数仍未找到满足要求的扰动，返回最后得到的新张量
    return perturbed_dict

# def compute_similarity(dict1, dict2):
#     # 这里使用欧式距离作为相似性度量，你可以根据需要选择其他度量方式
#     for key in dict1.keys():

    

#     return similarity

def narrowingDistance(cur_model, target_model, upper_distance):

    std_keys = get_key_list(cur_model.state_dict().keys())
    # 定义欧几里得距离的目标范围
    target_distance = upper_distance
    cur_state_dict = cur_model.state_dict()
    target_state_dict = target_model.state_dict()

    # 迭代优化过程
    scale_factor = 0.95
    for step in range(100):
        distance = model_dist_norm(cur_state_dict, target_state_dict)
        if distance < upper_distance:
            break
        
        for key in std_keys:
            tmp_tensor = cur_state_dict[key] + scale_factor * (target_state_dict[key] - cur_state_dict[key])
            cur_state_dict[key] = tmp_tensor
    
    return cur_state_dict





def dict2gradient(model_dict, std_keys):
    """
    将梯度展开成为一维张量
    """
    for idx, key in enumerate(std_keys):
        if idx == 0:
            grads = model_dict[key].view(-1)
        grads = torch.cat((grads, model_dict[key].view(-1)), dim = 0)

    print("grads shape is ", grads.shape)

    return grads


def gradient2dict(weights, std_keys, std_dict):
    # 重构张量，重构字典 
    update_dict = {}
    front_idx = 0
    end_idx = 0
    # mal_update张量重构
    for k in std_keys:
        tmp_len = len(list(std_dict[k].reshape(-1)))
        end_idx = front_idx + tmp_len
        tmp_tensor = weights[front_idx:end_idx].view(std_dict[k].shape)
        update_dict[k] = tmp_tensor.clone()
        front_idx = end_idx
    return update_dict


def computeTargetDistance(model_dicts, global_model, ratio):
    """
    计算在目标范围内的恶意用户聚合全局模型的欧氏距离的门槛值

    Args:
        benign_models(list): 同一分组内良性用户的集合
        global_model(model): 全局模型
        ration(float):希望返回的distance的排序比例
    Returns:
        target_distance: 返回目标位置的model距离全局模型的distance
    
    """
    res_distance = []

    print("len of model dicts is ", len(model_dicts))

    for model_dict in model_dicts:
        tmp_distance = model_dist_norm(model_dict, global_model.state_dict())
        tmp_similarity = cal_similarity(model_dict, global_model.state_dict())
        res_distance.append(tmp_distance)
        print("compute distance is ", tmp_distance)
        print("compute similarity is ", tmp_similarity)
    res_distance.sort()

    max_idx = int(len(model_dicts)) - 1

    target_distance = (res_distance[max_idx] - res_distance[0]) * ratio + res_distance[0]

    return target_distance


def modelAvg(benign_model_dicts, num_attacker, malicious_model):

    keys = benign_model_dicts[0].keys()
    avg_dict = copy.deepcopy(benign_model_dicts[0])

    for key in keys:
        tmp_param = []

        for model_dict in benign_model_dicts:
            tmp_param = model_dict[key].clone().unsqueeze(0) if len(tmp_param) == 0 else torch.cat((tmp_param, model_dict[key].clone().unsqueeze(0)), 0)

        if num_attacker != 0:
            for i in range(num_attacker):
                tmp_param = torch.cat((tmp_param, malicious_model.state_dict()[key].clone().unsqueeze(0)), 0)
        
        avg_param = torch.mean(tmp_param, dim = 0)
        avg_dict[key] = avg_param

    return avg_dict