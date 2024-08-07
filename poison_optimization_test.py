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

initial_w_rand = None

def cal_similarity(ori_params, mod_params):
    std_keys = get_key_list(ori_params.keys())
    params1 = torch.cat([ori_params[k].view(-1) for k in std_keys])
    params2 = torch.cat([mod_params[k].view(-1) for k in std_keys])

    # 计算余弦相似度
    cos_similarity = F.cosine_similarity(params1, params2, dim=0)

    return cos_similarity

def  model_dist_norm(ori_params, mod_params):
    squared_sum = 0
    distance1 = 0
    distance2 = 0

    pdlist = nn.PairwiseDistance(p=2)
    # 遍历参数字典，计算差异的平方和
    for key in ori_params.keys():
        # print("key is ",key)

        if key.endswith('num_batches_tracked'):  
            continue  
        t1 = ori_params[key]
        t2 = mod_params[key]
    # # 确保t1和t2都是Tensor，并且可以进行unsqueeze
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if ori_params[key].ndimension() == 1:
                t1 = ori_params[key].unsqueeze(0)
                t2 = mod_params[key].unsqueeze(0)

            else:
                t1 = ori_params[key]
                t2 = mod_params[key]
            # print("shape t1 = ",t1.shape)
            # print("shape t2 = ",t2.shape)
            temp1 = torch.sum(pdlist(t1 ,t2))
            output = torch.sum(temp1)
            distance1 += output
            # diff = torch.subtract(ori_params[key], mod_params[key])
            # distance2 += torch.norm(diff,p=2)  # 取平方并累加到distance中 
            # distance2 += torch.norm(diff.float()) 
    # print("distance 1 is",distance1)
    print("distance x is",distance1)
    return distance1


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
def Outline_Poisoning(args, global_model, malicious_models, train_dataset, distance_threshold, pinned_accuracy_threshold,w_rand):
    # global initial_w_rand
    ref_model = global_model 
    new_distance_ratio = 2.0 
    # distance_threshold = cal_ref_distance(malicious_models, ref_model, new_distance_ratio) # 计算参考L2 distance 门槛 
    # print("calculated distance threshold is ", distance_threshold)
    # return w_rand
    ###
    w_poison, optimization_res = phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold,  0.8)
    # 如果没有成功优化，则下一轮的distance不应该被改变
    if not optimization_res:
        # distance ratio 不应该被改变
        print("optimization failed")
    else:
        return w_poison

    # print("new distance ratio is", new_distance_ratio)
    return w_poison 

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

def cal_Norm(model_dict):
    res = 0
    for key in model_dict.keys():
        res += torch.norm(model_dict[key].double())
    return res

def cal_ref_distance(malicious_models, global_model, distance_ratio):
    
    distance_res = computeTargetDistance(malicious_models, global_model, distance_ratio)

    

    return distance_res


"""
自蒸馏 + adpative clipping
得到符合要求的投毒结果
"""
def phased_optimization(args, global_model, w_rand, train_dataset, distance_threshold, pinned_accuracy_threshold):

    # parameter determination
    round = 0
    MAX_ROUND = 3
    entropy_threshold = 1
    
     # 准备教师模型
    teacher_model = copy.deepcopy(global_model)
    # 准备学生模型 
    student_model = copy.deepcopy(global_model)
    test_model = copy.deepcopy(global_model)

    teacher_model.load_state_dict(w_rand)
    student_model.load_state_dict(w_rand)
    teacher_model.eval()
    
    distillation_res = True
    round = 0

    while round < MAX_ROUND:
        
        test_acc, test_loss, test_entropy = test_inference(args, copy.deepcopy(student_model), train_dataset)

        student_model.load_state_dict(w_rand)
        test_distance = model_dist_norm( global_model.state_dict(),w_rand)
        test_distance_1 = model_dist_norm( global_model.state_dict(),student_model.state_dict())
        w_semi = Avg(global_model.state_dict(),w_rand)
        # test_model.load_state_dict(w_semi)
        test_acc_1, loss_1,entropy_1 = test_inference(args, test_model, train_dataset)
        test_simliarity = cal_similarity( global_model.state_dict(),w_rand)
        print("________________________________")
        print("test distance is ", test_distance)
        print("test distance1 is ", test_distance_1)
        print("test similarity is ", test_simliarity)
        print("test acc is ", test_acc)
        print("test loss is", test_loss)
        print("test entropy is ", test_entropy)
        print("________________________________")
        
        if test_distance <= distance_threshold and test_acc_1 <= pinned_accuracy_threshold and test_entropy  <= entropy_threshold and test_loss <=2:
            return w_rand, True
        elif test_distance > distance_threshold:
            w_rand = adaptive_scaling(w_rand, global_model.state_dict(), distance_threshold, test_distance)
            # student_model.load_state_dict(w_rand_1)
            round = round - 1
              
        elif (test_entropy > entropy_threshold or test_loss > 2) and distillation_res == True:
            if test_loss > 1:
                print("0")
                distillation_res, w_rand = self_distillation(args,teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
            else:
                print("1")
                distillation_res, w_rand = self_distillation(args,teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
            # distillation_res, w_rand = self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold,  distance_threshold, distillation_round = 10)
        
        
        
        else:
            print("2")
            test_distance_2 = model_dist_norm( global_model.state_dict(),student_model.state_dict())
            print("test distance 2 is ", test_distance_2)
            distillation_res, w_rand = self_distillation(args,  teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 5)
            
        student_model.load_state_dict(w_rand)
        round = round +1

    test_acc, test_loss, test_entropy = test_inference(args, student_model, train_dataset)
    test_distance = model_dist_norm(student_model.state_dict(), global_model.state_dict())
    w_semi = Avg(global_model.state_dict(),student_model.state_dict())
    test_model.load_state_dict(w_semi)
    test_acc_1, loss_1,entropy_1= test_inference(args, test_model, train_dataset)
    test_simliarity = cal_similarity(student_model.state_dict(), global_model.state_dict())
    print("________________________________")
    print("test distance is ", test_distance)
    print("test similarity is ", test_simliarity)
    print("test acc is ", test_acc_1)
    print("test loss is", test_loss)
    print("test entropy is ", test_entropy)
    print("________________________________")
    print("final entropy")
    if test_entropy> 1:
        distillation_res, w_rand = self_distillation(args,  teacher_model, student_model, train_dataset, entropy_threshold, global_model, pinned_accuracy_threshold, distance_threshold, distillation_round = 1)

    # 返回值为模型参数
    print("reach max round") # accuracy在这里调整
    return w_rand, False


"""
接入两种方案进行尝试
"""
def adaptive_scaling(w_rand, ref_model_dict, distance_threshold, test_distance):
    use_case = 5
  
    if use_case == 5:
        pdlist= nn.PairwiseDistance(p=2)
        cal_distance = test_distance
        while cal_distance > distance_threshold:
            ratio = math.sqrt((distance_threshold / cal_distance)) * 0.98 # 
            print("cal ratio is ",ratio)
            keys = reversed(get_key_list(ref_model_dict))
            for key in keys:
                w_rand[key] = torch.sub(w_rand[key], ref_model_dict[key]) * ratio + ref_model_dict[key]
            cal_distance = model_dist_norm(w_rand, ref_model_dict)
            print("cal_distance is ", cal_distance)
        return_distance = model_dist_norm(w_rand, ref_model_dict)
        print("return_distance is ", return_distance)
        return w_rand
    
    return w_rand

def Avg(ref_state_dict, mal_state_dict):
    res_dict = copy.deepcopy(ref_state_dict)
    for key in ref_state_dict.keys():
        res_dict[key] = 2 * ref_state_dict[key] + 2 * mal_state_dict[key]
        
    return res_dict

def self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, accuracy_threshold, distance_threshold,  distillation_round):
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
    
    test_model = copy.deepcopy(ref_model)

    teacher_model.to(device)
    student_model.to(device)
    
    num_epochs = distillation_round
    temperature = 50 ### 增大
    alpha = 0.88
    beta = 0.12
    previous_loss = 0

    student_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        acc, loss, avg_entropy = test_inference(args, copy.deepcopy(student_model), train_dataset)

        w_semi = Avg(ref_model.state_dict(),student_model.state_dict())
        test_model.load_state_dict(w_semi)
        acc_1, loss_1,entropy_1 = test_inference(args, test_model, train_dataset)
        compute_distance = model_dist_norm( ref_model.state_dict(),student_model.state_dict())
        compute_norm  = cal_Norm(student_model.state_dict())
        cos_sim = cal_similarity(ref_model.state_dict(),student_model.state_dict())

        # loss =1
        if epoch != 0:
            # 防止循环太多次
            if abs(loss - previous_loss) < 0.005:
                print("exit early")
                return False, student_model.state_dict()
        previous_loss =  loss
        

        print("++++++++++++++++++++")
        print("after training ",epoch)
        print("avg test entropy is ", avg_entropy)
        print("avg test loss is ", loss)
        print("avg accuracy is ", acc_1) 
        print("solo acc is ",acc)
        print("compute similarity  is ",cos_sim)
        print("compute_distance is ",compute_distance)
        print("compute_norm is ",compute_norm)
        print("++++++++++++++++++++")

        # acc_1 =0.8
        # avg_entropy = 1.1


        if avg_entropy <= entropy_threshold and acc_1<= accuracy_threshold and loss <= 2:
        # if loss <=2 and avg_entropy <= entropy_threshold :
            return True, student_model.state_dict()
        elif avg_entropy <= entropy_threshold and loss > 2:
            print("change alpha")
            alpha = 0.7
            beta = 0.3
            # alpha = 0.5
            # beta = 0.4


        # elif 
        elif acc_1<= accuracy_threshold:
            #  loss <= 1:  #0.7 0.3 for fmnist
            print("restore alpha")
            # alpha = 0.7
            # beta = 0.3 #0.88 0.12 for fmnist
            alpha = 0.8
            beta = 0.2
        else:
            print("distillation for acc")
            # alpha = 0.9 # 改了0.9  ——0422
            # beta = 0.1#0.88 0.12 for fmnist
            alpha = 0.8 # 改了0.9  ——0422
            beta = 0.2#
        #     w_seed = student_model.state_dict()
        #     for key in w_seed.keys():
        #         w_seed[key] = -w_seed[key]
        #     student_model.load_state_dict(w_seed)
        

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_model.zero_grad()
            # 修改点1：设置模型参数需要梯度
            for param in student_model.parameters():
                param.requires_grad_(True)

            teacher_labels = list()
            _ , teacher_outputs,PLR = teacher_model(images)
            for item in teacher_outputs:
                pred_label = int(torch.max(item, 0)[1])
                teacher_labels.append(pred_label)

            pred_is = torch.tensor(teacher_labels)
            pred_is = pred_is.to(device)
            stu_out, student_outputs,PLR = student_model(images)
            # ref_out, ref_outputs, PLR= ref_model(images)
            _ , teacher_outputs,PLR= teacher_model(images)
            l_loss = criterion1(stu_out, pred_is)
            t_loss = criterion1(stu_out,  labels) #  增加了正常的loss
            loss =  alpha * l_loss + beta * t_loss

            # loss = l_loss
            
            loss.backward()
            optimizer.step(list(student_model.parameters()))
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
    

def sign_flip(orginal_dict):
    for key in orginal_dict.keys():
        orginal_dict[key] = -1 * orginal_dict[key]
    return orginal_dict

def add_small_perturbation(original_model, args, pinned_accuracy, train_dataset, distance_threshold, perturbation_range):
    """
    在原有张量上添加较小的扰动，使得新生成的张量在
    1. 欧氏距离
    与原向量保证一定的相似度
    """
    # std_keys = original_model.state_dict().keys()
    correct,total = 0.0,0.0
    test_model= copy.deepcopy(original_model)
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr, weight_decay=1e-4)
    # optimizer = MyPOptimizer(test_model.parameters(),lr=args.lr)
    for round in range(15):
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            test_model.zero_grad()
            # 修改点1：设置模型参数需要梯度
            for param in test_model.parameters():
                param.requires_grad_(True)


            output, out,PLR= test_model(images)
            # # 构造[batches,categaries]的真实分布向量
            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            pred_dec = torch.eq(pred_labels, labels)
            current_acc = torch.sum(pred_dec).item() + 1e-8
            correct += current_acc
            total += len(labels)
            # categaries = output.shape[1]
            loss = -1 *  criterion(output, labels)
            loss.backward()
            optimizer.step()

        accuracy  = correct/total
        print("batch acc is",   accuracy )

    return test_model.state_dict()

def cal_inner_product(local_dict, previous_local_dict, mal_dict):
    keys = get_key_list(local_dict.keys())
    res = 0
    for key in keys:
        mal_tensor = mal_dict[key].to(local_dict[key].dtype)
        diff1 = (mal_tensor - previous_local_dict[key]).view(-1)
        diff2 = (local_dict[key] - previous_local_dict[key]).view(-1)
        temp = torch.dot(diff1, diff2)
        res = res + temp
    
    return res

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
        tmp_norm =cal_Norm(model_dict)
        res_distance.append(tmp_distance)
        print("compute distance is ", tmp_distance)
        print("compute norm is ", tmp_norm)
    res_distance.sort()

    max_idx = int(len(model_dicts))-1

    target_distance = res_distance[max_idx] 

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