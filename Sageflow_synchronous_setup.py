#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from options import args_parser
import torch

import sys

sys.path.append('./drive/My Drive/federated_learning') # 添加路径至sys.path

from update import LocalUpdate, test_inference, DatasetSplit
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from utils1 import *
from added_funcs import poison_Mean, scale_attack
from resnet import *
from otherGroupingMethod import *
import csv
import os


# For experiments with only adversaries

if __name__ == '__main__':

    args = args_parser()

    start_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    path_project = os.path.abspath('..')

    exp_details(args) # 定义在utils里的函数，用于将实验参数打印出来
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    train_dataset, test_dataset, (user_groups, dict_common) = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)

        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)

        elif args.dataset == 'cifar':

            if args.detail_model == 'simplecnn':
                global_model = CNNCifar(args=args)
            elif args.detail_model == 'vgg':
                global_model = VGGCifar()
            elif args.detail_model == 'resnet':
                global_model = ResNet18()


    elif args.model == 'MLP':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    final_test_acc = []
    total_filter_num = 0
    print_every = 2
    val_loss_pre, counter = 0, 0
    
    # 增加权重
    users_weights = np.random.randint(10, size=args.num_users)
    
    prob = np.array(users_weights) / np.sum(users_weights)

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)
        
        idxs_users = np.random.choice(range(args.num_users), size = m , p = prob , replace=False)
        
        print("m is ",m)


        n = int(m*args.attack_ratio)
        attack_users = np.random.choice(idxs_users, n , replace=False)

        loss_on_public = []
        entropy_on_public = []
        global_weights_rep = copy.deepcopy(global_model.state_dict())
        # print(global_weights_rep.keys())

        for idx in idxs_users:

            if idx in attack_users and args.data_poison==True:

                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], data_poison=True,idx=idx)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], data_poison=False, idx=idx)

            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )


            if idx in attack_users and args.model_poison ==True:
                # w = sign_attack(w, args.model_poison_scale)
                w = scale_attack(global_model.state_dict(), w, args.scale_weight, n)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            global_model.load_state_dict(w)

            # Compute the loss and entropy for each device on public dataset
            common_acc, common_loss, common_entropy = test_inference(args, global_model, DatasetSplit(train_dataset, dict_common))
            loss_on_public.append(common_loss)
            entropy_on_public.append(common_entropy)

            global_model.load_state_dict(global_weights_rep)


        if args.new_poison == True:
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            std_keys = std_dict.keys()
            tensor_params = list()
            param_updates = list()
            param_update = list()
            for update_item in local_weights:
                param_new = []
                for key in std_keys:
                    param_new.append(copy.deepcopy(update_item[key]))
                    # print("update_item[key] is",update_item[key].size())
                param_update = [] # 清空
                for i in range(len(param_new)):
                    sub_res = torch.sub(param_new[i], 0).reshape(-1)
                    param_update = sub_res if len(param_update) == 0 else torch.cat((param_update, sub_res), 0)
            # print("param_update size is ",param_update.size())
                param_updates = param_update.clone().unsqueeze(0) if len(param_updates) == 0 else torch.cat((param_updates, param_update.clone().unsqueeze(0)), dim=0)  # 先用unsqueeze(0)增加维度
            
            # print("param_updates size is ",param_updates.size())
            
            avg_update = torch.mean(param_updates, 0) # 计算平均值
            mal_update = poison_Mean(param_updates, avg_update, args, m, m-n)
            
            # 重构张量，重构字典 
            mal_dict = {}
            front_idx = 0
            end_idx = 0
            # mal_update张量重构
            for k in std_dict.keys():
                tmp_len = len(list(std_dict[k].reshape(-1)))
                # print("tmp_len is", tmp_len)
                # print("kth tensor size  is", std_dict[k].size())
                end_idx = front_idx + tmp_len
                tmp_tensor = mal_update[front_idx:end_idx].view(std_dict[k].shape)
                mal_dict[k] = copy.deepcopy(tmp_tensor)
                front_idx = end_idx
            # print(mal_dict.keys())
            # 重新计算common_acc, common_loss, common_entropy
            global_model.load_state_dict(mal_dict) # 加载恶意的梯度
            mal_common_acc, mal_common_loss, mal_common_entropy = test_inference(args, global_model, DatasetSplit(train_dataset, dict_common))
            global_model.load_state_dict(global_weights_rep)
            
            for client_num in range(m):
                if client_num >= m-n:  #将恶意用户的梯度替换成上面构造的恶意梯度，这里是选取最后几个用户作为恶意用户
                    local_weights[client_num] = copy.deepcopy(mal_dict) #在这个地方不
                    loss_on_public[client_num] = mal_common_loss # 更新loss_on_public 列表
                    entropy_on_public[client_num] = mal_common_entropy # 更新entropy_on_public列表
            
        if args.update_rule == 'Sageflow':
            # Averaging local weights via entropy-based filtering and loss-wegithed averaging
            global_weights, no_use_len1= Eflow(local_weights, loss_on_public,entropy_on_public,epoch)
            # print("filter num is ", filter_num)
            # total_filter_num = total_filter_num + filter_num
        elif args.update_rule == 'Krum':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            std_keys = std_dict.keys()
            user_num = len(list(local_weights))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = modifyWeight(std_keys, local_weights)
            global_update, _ = Krum(weight_updates, user_num - attacker_num)
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, global_update)
        elif args.update_rule == 'Trimmed_mean':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            std_keys = std_dict.keys()
            user_num = len(list(local_weights))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = modifyWeight(std_keys, local_weights)
            global_update = Trimmed_mean(weight_updates, attacker_num)
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, global_update)
        else:
            global_weights = average_weights(local_weights)

        # Update global weights
        for k,v in global_weights.items():
            print(k) #只打印key值，不打印具体参数。
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)  # epoch loss in server

        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()

        for c in range(args.num_users):
            if c in attack_users and args.data_poison == True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=True, idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, idx=c)


            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(train_test_acc)

        if (epoch + 1) % 1 == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss , _= test_inference(args, global_model, test_dataset)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
        final_test_acc.append(test_acc)

    # print("total filter num is ", total_filter_num)
    print(f' \n Results after {args.epochs} global rounds of training:')

    # Averaging test accuarcy across device which has each test data inside
    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Final Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))

    # Final test accuarcy for global test dataset.
    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))

    exp_details(args)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    if args.data_poison == True:
        attack_type = 'data'
    elif args.model_poison == True:
        attack_type = 'model'
        model_scale = '_scale_' + str(args.model_poison_scale)
        attack_type += model_scale
    else:
        attack_type = 'no_attack'
    # 下面这一行代码有问题
    file_n = f'accuracy_sync_{args.update_rule}_{args.dataset}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.seed}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()












