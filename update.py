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
def test_inference(args, model, test_dataset, global_model):

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
            # real_distribute = []
            # for l in labels:
            #     temp = [0] * categaries
            #     idx = int(l)
            #     temp[idx] = 1
            #     real_distribute.append(copy.deepcopy(temp))
            # real_distribute = torch.tensor(real_distribute).float().to(device)
            # log_pre = F.softmax(out, dim=1)
            # real_distribute = F.log_softmax(real_distribute, dim =1)
            # KL_loss = F.kl_div(real_distribute, log_pre, reduction='batchmean') 
            # log_pre = F.log_softmax(out, dim=1)
            # real_distribute = F.softmax(real_distribute, dim =1)
            # KL_loss = F.kl_div(log_pre, real_distribute,  reduction='batchmean') 
            # print("out is ", out)
            # labels_test = F.softmax(labels_test,dim = 0)
            
            # print("KL_loss is ", KL_loss)
            # batch_KL.append(KL_loss)
            # print("output is ", F.softmax(out, dim=1))
            # print("out is ", out)
            
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
            

        distance = model_dist_norm(global_model.state_dict(), model.state_dict())
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


    # 遍历参数字典，计算差异的平方和
    for key in ori_params.keys():
        squared_sum += torch.sum(torch.pow(ori_params[key] - mod_params[key], 2))

    # 计算平方和的平方根，即模型参数之间的距离
    distance = math.sqrt(squared_sum)
    return distance


def modifyLabel(args, model, train_dataset, global_model):
    # testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    # allres = []

   
    # for batch_idx, (images, labels) in enumerate(testloader):
    #     images, labels = images.to(device), labels.to(device)
    #     output, out = global_model(images) # 看一下test时的输出

    model.eval()
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    allres = {}
    for i in range(11):
        allres[i] = [] # 用来存储每个label类别的张量

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            output, out = model(images)
            # 对out结果 做一个平均
            # print("out is ",out)
            for i in range(len(labels)):
                # print("current i is ",i)
                # print("current label is ",int(labels[i]))

                # print("current out is ",out)
                allres[int(labels[i])].append(out[i])

            # result = torch.mean(out, dim = 0)
            # pred_labels = torch.max(out, -1)
            # pred_not = torch.min(out, -1)
            # print("result is ", pred_not)
            # print("label is ", type(labels))
            # allres.append(result)
    
    labelmap = {}
    for j in range(1,11):
        labelmap[j] = 1
        
    for i in range(1,11):
        count = 0
        tmp = allres[i]
        for item in tmp:
            if count == 0:
                param_out = item.unsqueeze(0)
                count = 1
                continue
            param_out = torch.cat((param_out, item.unsqueeze(0)), dim = 0)

        param_out = torch.tensor(param_out)
        res = torch.mean(param_out, dim = 0)
        pred_not = int(torch.min(res, 0)[1])
        pred_is = int(torch.max(res, 0)[1])
        # print("current label is " , i)
        # print("predict  is ", pred_is)
        # print("predict not is ",pred_not)
        # labelmap[i] = pred_not

        ## ##取第二大的值
        max_idx = int(torch.max(res, 0)[1])
        res[max_idx] = 0
        max_2_idx = int(torch.max(res, 0)[1])
        labelmap[i] = max_2_idx


            ## 接下来按照predict not 对label进行修改
    print("labelmap is ", labelmap)
    return labelmap


def self_distillation(model, args, train_dataset, benign_models, num_attacker):
    """
    自蒸馏随机恶意梯度
    """
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    b4_posion_dict = modelAvg(benign_models, num_attacker = 0, malicious_model = None)
    ref_distance = computeTargetDistance(benign_models, model, ratio = 0.8)
    print("ref distance is ", ref_distance)
    w_rand = add_small_perturbation(model, args, ref_distance)


    # print("after is ")
    # print(dict2gradient(w_rand, std_keys))

    # 准备教师模型
    teacher_model = CNNMnist(args=args)
    teacher_model.load_state_dict(w_rand) # 加载随机梯度
    
    teacher_model.eval() # eval函数究竟代表什么


    # 准备学生模型
    student_model = CNNMnist(args=args)
    student_model.load_state_dict(w_rand)

    # 定义优化器
    lr = args.lr 
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion1 = nn.NLLLoss().to(device)
    teacher_model.to(device)
    student_model.to(device)


    acc,avg_loss, avg_entropy = test_inference(args, teacher_model, train_dataset, model)
    distance = model_dist_norm(model.state_dict(), teacher_model.state_dict())
    print("xxxxxxxx  teacher model  xxxxxxxxxxxxxx")
    print("avg test entropy is ", avg_entropy)
    print("avg test loss is ", avg_loss)
    print("avg distance is ", distance)
    print("avg accuracy is ", acc)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    
    sorted_distance = computeTargetDistance(benign_models, model, ratio = 0.8)

    # 模型训练
    num_epochs = 20
    student_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            student_model.zero_grad()
            teacher_labels = list()
            _ , teacher_outputs = teacher_model(images)
            for item in teacher_outputs:
                pred_label = int(torch.max(item, 0)[1])
                # print("pred_label is ",pred_label)
                teacher_labels.append(pred_label)
            pred_is = torch.tensor(teacher_labels)
            pred_is = pred_is.to(device)

            #######
            


            ####

            stu_out, student_outputs = student_model(images)

            
            # loss_1 = distillation_loss(student_outputs, teacher_outputs, temperature = 10.0)
            loss_1 = criterion1(stu_out, pred_is)
            
            loss =  loss_1
                                                                                                                                              
            # print("distillation loss is ", loss)
            loss.backward()
            optimizer.step()

        if epoch%4 == 0:
            acc,avg_loss, avg_entropy = test_inference(args, student_model, train_dataset, model)
            distance = model_dist_norm(model.state_dict(), student_model.state_dict())
            print("++++++++++++++++++++")
            print("after training ",epoch)
            print("before scale ")
            print("avg test entropy is ", avg_entropy)
            print("avg test loss is ", avg_loss)
            print("avg distance is ", distance)
            print("avg accuracy is ", acc)
            print("++++++++++++++++++++")

            alpha = 0.5
            # C = 5.0
            beta = 0.2
            cur_distance = model_dist_norm(student_model.state_dict(), model.state_dict())
            # 计算 distance是否与良性距离在一定范围内，如果超出该范围，则再次进行放缩
            # Loss = 自蒸馏Loss + alpha*[D"同一个陈旧度内的"(恶意模型梯度， 参考模型梯度) - C * 第2C大的良性梯度与参考模型之间的Distance] - beta * D(投毒前全局模型，投毒后全局模型)
            
            pre_post_distance = model_dist_norm(modelAvg(benign_models, num_attacker, student_model), model.state_dict())
            
                    # 1. benign  -> weighted_average ->  当前轮的聚合结果 - > 参考模型梯度
                    # 2. weighted_average
            
            constrain_distance = alpha * sorted_distance + beta * pre_post_distance  ### 本轮的更新情况 1-5   1 2 4 -> 
            print("____________________________________")
            print("cur_distance is ", cur_distance)
            print("sorted_distance is ", sorted_distance)
            print("pre_post_distance is ", pre_post_distance)
            print("constrain_distance is ",constrain_distance)
            print("____________________________________")

            if cur_distance > constrain_distance:
                w_rand = narrowingDistance(student_model, model, upper_distance = constrain_distance)
                student_model.load_state_dict(w_rand)

            acc,avg_loss, avg_entropy = test_inference(args, student_model, train_dataset, model)
            distance = model_dist_norm(model.state_dict(), student_model.state_dict())
            print("++++++++++++++++++++")   
            print("after training ",epoch)
            print("after scale ")
            print("avg test entropy is ", avg_entropy)
            print("avg test loss is ", avg_loss)
            print("avg distance is ", distance)
            print("avg accuracy is ", acc)
            print("++++++++++++++++++++")
            
            
        teacher_model.load_state_dict(student_model.state_dict()) # 这一步是不是必须的

    print("finish training-----------------")
    # 测试恶意梯度的熵值
    acc,avg_loss, avg_entropy = test_inference(args, student_model, train_dataset, model)
    distance = model_dist_norm(model.state_dict(), student_model.state_dict())
    print("++++++++++++++++++++")
    print("final test entropy is ", avg_entropy)
    print("final test loss is ", avg_loss)
    print("final distance is ", distance)
    print("++++++++++++++++++++")

    return student_model.state_dict()

def distillation_loss(student_outputs, teacher_outputs, temperature):
    """
    自蒸馏的损失函数

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
    loss = F.kl_div(stu_s, teacher_s, size_average=False) * (temperature ** 2) / student_outputs.shape[0]
    
    return loss
    

def add_small_perturbation(original_model, args, target_distance, perturbation_range=(-0.01, 0.01)):
    """
    在原有张量上添加较小的扰动，使得新生成的张量在
    1. 欧氏距离
    与原向量保证一定的相似度
    """
    # std_keys = original_model.state_dict().keys()

    orignal_state_dict = original_model.state_dict()
    perturbed_dict = copy.deepcopy(orignal_state_dict)
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'

    max_iterations = 100
    iteration = 0
    while iteration < max_iterations:
        # 计算当前扰动范围的中点
        mid_min = perturbation_range[0]  / 2.0
        mid_max = mid_min * -1

        # 生成中点扰动张量
        
        for key in orignal_state_dict.keys():
            temp_original = orignal_state_dict[key]
            perturbs = torch.tensor(np.random.uniform(low=mid_min, high=mid_max, size=temp_original.shape)).to(device)
            perturbs = torch.add(perturbs, temp_original)
            perturbed_dict[key] = perturbs

        # 计算新张量和原张量之间的相似性
        distance = model_dist_norm(orignal_state_dict, perturbed_dict)
        print("====================")
        print("iteration is ", iteration)
        print("distance is ", distance)
        print("====================")

        # 判断是否达到目标相似性
        if distance <= target_distance :
            # 相似性满足要求，返回新张量
            # print("perturbation shape is ", perturbed_tensor.shape)
            # print(perturbed_tensor)
            return perturbed_dict
        else:
            # 相似性不满足要求，更新扰动范围，继续迭代
            perturbation_range = (mid_min, mid_max)
            iteration += 1

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


def computeTargetDistance(benign_model_dicts, global_model, ratio):
    """
    计算在目标范围内的良性用户聚全局模型的欧氏距离的门槛值

    Args:
        benign_models(list): 同一分组内良性用户的集合
        global_model(model): 全局模型
        ration(float):希望返回的distance的排序比例
    Returns:
        target_distance: 返回目标位置的model距离全局模型的distance
    
    """
    res_distance = []

    for model_dict in benign_model_dicts:
        tmp_distance = model_dist_norm(model_dict, global_model.state_dict())
        res_distance.append(tmp_distance)
    res_distance.sort()

    idx = int(ratio * len(benign_model_dicts)) - 1

    return res_distance[idx]


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