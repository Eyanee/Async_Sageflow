import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy
import math
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from customLossFuncs import CustomDistance1
from otherGroupingMethod import get_key_list



class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, idx,data_poison, inverse_poison, labelmap = {}, delay=False):
        self.args = args
        self.idx= idx
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'

        self.criterion = nn.NLLLoss().to(self.device)
        self.delay = delay
        self.data_poison = data_poison
        self.inverse_poison = inverse_poison
        self.labelmap = labelmap

    def train_val_test(self, dataset, idxs):

        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=max(int(len(idxs_test)/10),1), shuffle=False)

        return trainloader, testloader

# 修改这一部分的梯度更新方式
    def update_weights(self,model, global_round):
        model.train()
        epoch_loss = []


        if self.args.optimizer == 'sgd':

            lr = self.args.lr

            lr = lr * (0.5)**(global_round//self.args.lrdecay)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        elif self.args.optimizer == 'adam':

            lr = self.args.lr
            lr = lr * (0.5)**(global_round//self.args.lrdecay)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print("label is ", labels) # 在这个地方对label进行对应的反转 使用list对应下标进行修改
                if self.inverse_poison == True:
                    # 按照字典中对应的值进行修改
                    for i in range(1,11):
                        labels = torch.where(labels == i, self.labelmap[i], labels)
                    

                model.zero_grad()
                log_probs,_ = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs,_ = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct/total
        return accuracy, loss

# def test_inference(args, model, test_dataset):
def test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    batch_losses = []
    batch_entropy = []
    batch_KL = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            output, out = model(images)
            # 构造[batches,categaries]的真实分布向量
            categaries = output.shape[1]

            
            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            # print("Information is ", Information)
            
            entropy  = -1.0 * Information.sum(dim=1) # size [64]
            # print("entropy is ", entropy)
            average_entropy = entropy.mean().item()
            

            batch_loss = criterion(output, labels)
            batch_losses.append(batch_loss.item())

            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            # print("pred_labels is ", pred_labels)
            pred_dec = torch.eq(pred_labels, labels)
            current_acc = torch.sum(pred_dec).item() + 1e-8

            # print("average_entropy is ", average_entropy)
            batch_entropy.append(average_entropy)

            correct += current_acc
            total += len(labels)
            

        # distance = model_dist_norm(global_model.state_dict(), model.state_dict())
        accuracy  = correct/total
        # print("--------------------")
        # print("average entropy is ", sum(batch_entropy)/len(batch_entropy))
        # print("average loss is ", sum(batch_losses)/len(batch_losses))
        # print("average acc is ", accuracy)
        # print("average distance is ", distance)
        # print("--------------------")
        

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)


def model_dist_norm(ori_params, mod_params):
    squared_sum = 0
    distance = 0

    pdlist = nn.PairwiseDistance(p=2)
    # 遍历参数字典，计算差异的平方和
    for key in ori_params.keys():
        # squared_sum += torch.sum(torch.pow(ori_params[key] - mod_params[key], 2))
        # print("shape temp params is ", ori_params[key].shape)
        if ori_params[key].ndimension() == 1:
            t1 = ori_params[key].unsqueeze(0)
            t2 = mod_params[key].unsqueeze(0)
            # print("shape temp params is ", t1.shape)
        else:
            t1 = ori_params[key]
            t2 = mod_params[key]
        temp1 = pdlist(t1 ,t2)
        output = torch.sum(temp1)
        distance += output

    # 计算平方和的平方根，即模型参数之间的距离
    # distance = math.sqrt(squared_sum)
    return distance


def get_distance_list(ori_state_dict, mod_state_dict):
    squared_sum = 0
    distance = 0
    distance_list = []
    pdlist = nn.PairwiseDistance(p=2)
    # 遍历参数字典，计算差异的平方和
    for key in ori_state_dict.keys():
        # squared_sum += torch.sum(torch.pow(ori_params[key] - mod_params[key], 2))
        # print("shape temp params is ", ori_params[key].shape)
        if ori_state_dict[key].ndimension() == 1:
            t1 = ori_state_dict[key].unsqueeze(0)
            t2 = mod_state_dict[key].unsqueeze(0)
            # print("shape temp params is ", t1.shape)
        else:
            t1 = ori_state_dict[key]
            t2 = mod_state_dict[key]
        temp1 = pdlist(t1 ,t2)
        output = torch.sum(temp1)
        distance_list.append(output)

    return distance_list

    
"""
修改中
主要将benign models的参照换成 malicious model
改变 distance threshold的计算方式
"""

def phased_optimization(model, args, train_dataset, malicious_models, POISONED = True ):
    """
    分阶段优化函数
    """
    # parameter determination
    round = 0
    MAX_ROUND = 5
    entropy_threshold = 1.0
     # 准备教师模型
    teacher_model = copy.deepcopy(model)
    # 准备学生模型 
    student_model = copy.deepcopy(model)


    # initialization
    ref_model = copy.deepcopy(model) # 上一轮进行聚合的全局模型
    target_accuracy = 0.80 # 如何选择
    rank_ratio = 0.8 ## AFA是去掉20%的 consine similarity 离群值
    distance_threshold = computeTargetDistance(malicious_models, ref_model, rank_ratio)
    print(" distance_threshold is ", distance_threshold)
    
    w_rand = add_small_perturbation(ref_model, args, target_accuracy, train_dataset)
    teacher_model.load_state_dict(w_rand)
    student_model.load_state_dict(w_rand)
    teacher_model.eval()

    # outside loop1
    while True:
        compute_acc, compute_loss, compute_entropy = test_inference(args, student_model, train_dataset)
        compute_distance = model_dist_norm(student_model.state_dict(), ref_model.state_dict())
        print("++++++++++++++++++++")
        print("Round:  " ,round)
        print("compute_entropy is ", compute_entropy)
        print("compute_loss is ", compute_loss)
        print("compute_acc is ", compute_acc)
        print("compute_distance is ", compute_distance)
        print("++++++++++++++++++++")
        if compute_entropy <= entropy_threshold and compute_distance <= distance_threshold and compute_acc < target_accuracy:
            print("phased optimization succeed!")
            break 

        elif round > MAX_ROUND:
            # Reinitialize
            # distance_threshold 不变， 可以修改accuracy门槛 ————> 需要讨论
            print("reinitialization....")
            round = 0
            w_rand = add_small_perturbation(ref_model, args, target_accuracy, train_dataset)
            teacher_model.load_state_dict(w_rand)

        if compute_distance > distance_threshold:
            # scale_part
            # 直接按比例放缩
            print("scale....")
            w_scale = parameter_scaling(distance_threshold, compute_distance, student_model, ref_model)
            student_model.load_state_dict(w_scale)
        # 
        elif compute_entropy > entropy_threshold:
            # self_distillation part
            print("self_distillation....")
            res, w_distillation = self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, target_accuracy, distillation_round = 5)
            if res:
                student_model.load_state_dict(w_distillation)
            else:
                print("self_distillation reaches max round! go to reinitialization")
                round = MAX_ROUND
        
        else:
            res, w_distillation = self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, target_accuracy, distillation_round = 5)
            student_model.load_state_dict(w_distillation)
        round = round + 1 # 外循环计数


    # 返回值为模型参数
    return student_model.state_dict()


def self_distillation(args, teacher_model, student_model, train_dataset, entropy_threshold, ref_model, accuracy_threshold, distillation_round):
    """
    自蒸馏函数主体
    """
    ### 自蒸馏参数
    lr = args.lr * 0.001
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion1 = nn.NLLLoss().to(device)
    teacher_model.to(device)
    student_model.to(device)

    num_epochs = distillation_round
    temperature = 5
    alpha = 0.5
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
            # 正常情况下 20轮内可以将熵控制在门槛值以下
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
            _ , teacher_outputs = teacher_model(images)
            loss = alpha * criterion1(stu_out, pred_is)  + beta * temperature** 2 * soft_loss(student_outputs, teacher_outputs, temperature = 5)

            loss.backward()
            optimizer.step()

    # 设置False出口
    return True, student_model.state_dict()

def parameter_scaling(distance_threshold, compute_distance, student_model, ref_model):
    """
    分开scale
    """
    
    model_dict = student_model.state_dict()
    ref_dict = ref_model.state_dict()
    keys = get_key_list(model_dict.keys())
    
    ## 构造list
    ratio = (compute_distance - distance_threshold)/compute_distance
    # distance_list =  get_distance_list(ref_model.state_dict(), student_model.state_dict())
    while compute_distance > distance_threshold:
        ratio = (compute_distance - distance_threshold)/compute_distance
        print("ratio is ", ratio)
        for key in keys:
            diff  = torch.sub(ref_dict[key], model_dict[key])
            model_dict[key] = model_dict[key] + torch.mul(diff, ratio)
        compute_distance = model_dist_norm(model_dict, ref_dict)
        print("scaled distance is ",compute_distance)

    return model_dict

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
    

def add_small_perturbation(original_model, args, target_accuracy, train_dataset, perturbation_range=(-0.05, 0.05)):
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
    iteration = 0
    while iteration < max_iterations:
        # 计算当前扰动范围的中点
        mid_min = perturbation_range[0] * 1.1
        mid_max = mid_min * -1

        # 生成中点扰动张量
        for round in range(MAX_ROUND):
            for key in orignal_state_dict.keys():
                temp_original = orignal_state_dict[key]
                perturbs = torch.tensor(np.random.uniform(low=mid_min, high=mid_max, size=temp_original.shape)).to(device)
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
                
                return perturbed_dict
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
        res_distance.append(tmp_distance)
        print("compute distance is ", tmp_distance)
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