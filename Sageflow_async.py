#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from update import LocalUpdate, test_inference, DatasetSplit
from poison_optimization import Outline_Poisoning
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from resnet import *
from utils1 import *
from added_funcs import poison_Mean
import csv
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import os
from otherGroupingMethod import *


# For experiments with only stragglers
# For experiments with both stragglers and adversaries


if __name__ == '__main__':


    args = args_parser()

    start_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    path_project = os.path.abspath('..')

    exp_details(args)

    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')


    train_dataset, test_dataset, (user_groups, dict_common) = get_dataset(args) # 将（user_groups,dict_common）打包成元组，实际上dict_common的值为None

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
    pre_global_model = copy.deepcopy(global_model)

    global_weights = global_model.state_dict()

    train_accuracy = []
    final_test_acc = []
    print_every = 1

    pre_weights = {} 
    pre_index = {}
    # 多级staleness的存储
    for i in range(args.staleness + 1): 
        if i != 0:
            pre_weights[i] = []
            pre_index[i] = []

    # Device schedular
    scheduler = {}
    
    # Staleness 设定
    clientStaleness = {}
    
    # 目标Staleness设定
    TARGET_STALENESS  = 1

    # 投毒状态标志
    poisoned = False
    # 其他参数
    distance_ratio = 0.8
    adaptive_accuracy_threshold = 0.8
    pinned_accuracy_threshold = 0.8

 
    for l in range(args.num_users):
        scheduler[l] = 0
        clientStaleness[l] = 0

    global_epoch = 0
    
    all_users = np.arange(args.num_users)
    m = int(args.num_users * args.attack_ratio)
    n = args.num_users - m
    attack_users = all_users[-m:]
    print("attack user num is ",m)
    
    t = int(n/args.staleness)
    print(" t is ",t )
    
    # 为正常用户赋固定的Staleness值
    for i in range(args.staleness):
        front_idx = int(t * i)
        end_idx = front_idx + t
        for j in range(front_idx, end_idx):
            clientStaleness[j] = i + 1 
            
        
    print("attack_user is", attack_users)
    # 为恶意用户赋目标的Staleness值     
    for l in attack_users:
        clientStaleness[l] = TARGET_STALENESS   
            
    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        loss_on_public = {}
        entropy_on_public = {}
        local_index_delay = {}
        malicious_models = []

        for i in range(args.staleness + 1):
            loss_on_public[i] = []
            entropy_on_public[i] = []
            local_weights_delay[i] = []
            local_index_delay[i] = []


        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        global_weights_rep = copy.deepcopy(global_model.state_dict())

        # After round, each staleness group is adjusted
        local_delay_ew = copy.deepcopy(pre_weights[1])
        local_index_ew = copy.deepcopy(pre_index[1])

        for i in range(args.staleness):
            if i != 0:
                pre_weights[i] = copy.deepcopy(pre_weights[i+1])
                pre_index[i] = copy.deepcopy(pre_index[i+1])

        pre_weights[args.staleness] = [] # 对staleness的权重
        pre_index[args.staleness] = []

        ensure_1 = 0

        for idx in all_users:

            if scheduler[idx] == 0: # 当前轮次提交
                if idx in attack_users and args.data_poison == True: # 原 data_posion 入口
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=True, inverse_poison= False)

                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=False, inverse_poison= False)

                # Ensure each staleness group has at least one element
                scheduler[idx] = clientStaleness[idx]  # 重新赋值staleness
                print("current submit client idx and staleness is ", idx ,scheduler[idx])

            else:

                continue

            w, loss = local_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )
            if idx in attack_users and epoch > 29 and args.model_poison == True:
                malicious_models.append(w) #
                
            else:     
                ensure_1 += 1 # 平均分布
                
                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(w)

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, test_model,
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))
                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(w))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
            
        if idx in attack_users and epoch > 29 and args.model_poison == True:
            if not poisoned:
                poisoned = True
                pinned_accuracy_threshold = test_acc - 0.15# 上一轮全局模型的精度
                print("poisoned accuracy threshold is ",pinned_accuracy_threshold)
                adaptive_accuracy_threshold = pinned_accuracy_threshold
                malicious_dict, distance_ratio = Outline_Poisoning(args, pre_global_model, copy.deepcopy(global_model), malicious_models, train_dataset, distance_ratio, pinned_accuracy_threshold,
                                               adaptive_accuracy_threshold, False)
            else:
                malicious_dict, distance_ratio = Outline_Poisoning(args, pre_global_model, copy.deepcopy(global_model), malicious_models, train_dataset, distance_ratio, pinned_accuracy_threshold,
                                               adaptive_accuracy_threshold, True)

            test_model.load_state_dict(malicious_dict)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))
        
            for idx in attack_users:
                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(w))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)

            


        # 重新考虑PosionMean方式投毒的接入
        if epoch/TARGET_STALENESS == 0 and args.new_poison == True:
            std_dict = copy.deepcopy(global_weights)
            std_keys = get_key_list(std_dict.keys())
            ts = TARGET_STALENESS - 1
            param_updates = modifyWeight(std_keys, local_weights_delay[ts])
            avg_update = torch.mean(param_updates, 0) # 计算平均值
            user_num = len(list(local_weights_delay[ts]))
            attacker_num = int(len(attack_users)) #假设每一组内的attacker数量尽可能平均 
            mal_update = poison_Mean(param_updates, avg_update, args, user_num, user_num-attacker_num)
            
            # 重构张量，重构字典 
            mal_dict = restoreWeight(std_dict, mal_update)
            global_model.load_state_dict(mal_dict) # 加载恶意的梯度
            mal_common_acc, mal_common_loss, mal_common_entropy = test_inference(args, global_model, DatasetSplit(train_dataset, dict_common))
            global_model.load_state_dict(global_weights_rep)
                
            for client_num in range(user_num-attacker_num, user_num):
                if client_num >= user_num-attacker_num:  #将恶意用户的梯度替换成上面构造的恶意梯度，这里是选取最后几个用户作为恶意用户
                    local_weights_delay[ts][client_num] = copy.deepcopy(mal_dict) #在这个地方不
                    loss_on_public[ts][client_num] = mal_common_loss # 更新loss_on_public 列表
                    entropy_on_public[ts][client_num] = mal_common_entropy # 更新entropy_on_public列表
        

        for i in range(args.staleness):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    # Averaging delayed local weights via entropy-based filtering and loss-wegithed averaging
                    if len(local_weights_delay[i]) > 0:
                        print("current aggregation staleness i is ",i)
                        w_avg_delay, len_delay, num_attacker_1 = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        print("num1 of attacker is ",num_attacker_1)
                        pre_index[i].append(local_index_delay[i])
                elif args.update_rule == 'Krum':
                    # pre_weights[i].append({epoch: [average_weights(local_weights_delay[i]), len(local_weights_delay[i])]})
                    pre_weights[i].append(local_weights_delay[i]) # 修改为记录每一次的，不进行提前聚合, local_weights_delay[i]是List格式
                    pre_index[i].append(local_index_delay[i])
                elif args.update_rule == 'Trimmed_mean':
                    # pre_weights[i].append({epoch: [average_weights(local_weights_delay[i]), len(local_weights_delay[i])]})
                    pre_weights[i].append(local_weights_delay[i]) # 修改为记录每一次的，不进行提前聚合, local_weights_delay[i]是List格式
                    pre_index[i].append(local_index_delay[i])
                elif args.update_rule == 'AFA':
                    if len(local_weights_delay[i]) > 0:
                        # 在这一步将所有staleness != 1 的轮次进行对应的聚合
                        std_dict = copy.deepcopy(global_weights) # 标准字典值
                        std_keys = get_key_list(std_dict.keys())
                        # 调用AFA同时保留对应index
                        w_res, remain_index = pre_AFA(std_keys, local_weights_delay[i], local_index_delay[i], device)
                        w_avg = restoreWeight(std_dict, std_keys, w_res)
                        print("left index is ", remain_index)
                        len_delay = len(remain_index)
                        pre_weights[i].append({epoch: [w_avg, len_delay]})
                        pre_index[i].append(remain_index)
                elif args.update_rule == 'Bulyan':
                    pre_weights[i].append(local_weights_delay[i])
                    pre_index[i].append(local_index_delay[i])
                elif args.update_rule == 'Median':
                    pre_weights[i].append(local_weights_delay[i])
                    pre_index[i].append(local_index_delay[i])
                elif args.update_rule == 'Mean':
                    pre_weights[i].append(local_weights_delay[i])
                    pre_index[i].append(local_index_delay[i])
                else:
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append({epoch: [average_weights(local_weights_delay[i]), len(local_weights_delay[i])]})
                        pre_index[i].append(local_index_delay[i])

        if args.update_rule == 'Sageflow':
            # Averaging current local weights via entropy-based filtering and loss-wegithed averaging
            sync_weights, len_sync, num_attacker_2 = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)
            print("num2 of attacker is ",num_attacker_2)
            # Staleness-aware grouping
            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
        elif args.update_rule == 'Krum':
            # global_weights = Sag(epoch, average_weights(local_weights_delay[0]), len(local_weights_delay[0]),
            #                                      local_delay_ew, copy.deepcopy(global_weights))
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            user_num = len(list(local_weights_delay[0])) + len(list(local_delay_ew))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = preKrumGrouping(std_keys, copy.deepcopy(local_weights_delay[0]), local_delay_ew)# 第二个参数实际上是上一轮的pre_weights[1]
            global_update, _ = Krum(weight_updates, user_num - attacker_num)
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, std_keys, global_update)
            
        elif args.update_rule == 'Trimmed_mean':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            user_num = len(list(local_weights_delay[0])) + len(list(local_delay_ew))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = preKrumGrouping(std_keys, copy.deepcopy(local_weights_delay[0]), local_delay_ew)# 第二个参数实际上是上一轮的pre_weights[1]
            global_update= Trimmed_mean(weight_updates, attacker_num)
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, std_keys, global_update)
            
        elif args.update_rule == 'AFA':
            # 待做  ——————————  加入staleness aware grouping

            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            std_keys = get_key_list(std_dict.keys())
            w_res, remain_index = pre_AFA(std_keys, local_weights_delay[0], local_index_delay[0], device)
            w_avg = restoreWeight(std_dict, std_keys, w_res)
            print("left index is ", remain_index)
            global_weights = Sag(epoch, w_avg, len(remain_index), local_delay_ew, # local_delay_ew 实际上是上一轮的pre_weights[1]
                                                     copy.deepcopy(global_weights))
        
        elif args.update_rule == 'Bulyan':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            std_keys = get_key_list(std_dict.keys())
            user_num = len(list(local_weights_delay[0])) + len(list(local_delay_ew))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = preKrumGrouping(std_keys, copy.deepcopy(local_weights_delay[0]), local_delay_ew)# 第二个参数实际上是上一轮的pre_weights[1]
            global_update = Bulyan(weight_updates, user_num, user_num - attacker_num)
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, std_keys, global_update)
        elif args.update_rule == 'Median':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            std_keys = get_key_list(std_dict.keys())
            user_num = len(list(local_weights_delay[0])) + len(list(local_delay_ew))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = preKrumGrouping(std_keys, copy.deepcopy(local_weights_delay[0]), local_delay_ew)# 第二个参数实际上是上一轮的pre_weights[1]
            global_update = Median(weight_updates)
            print("global update type is", type(global_update))
            # 重新恢复dict
            global_weights = restoreWeight(std_dict, std_keys, global_update)    
        elif args.update_rule == 'Mean':
            std_dict = copy.deepcopy(global_weights) # 标准字典值
            # std_keys = std_dict.keys()
            std_keys = get_key_list(std_dict.keys())
            user_num = len(list(local_weights_delay[0])) + len(list(local_delay_ew))
            attacker_num = int(user_num * args.attack_ratio)
            weight_updates = preKrumGrouping(std_keys, copy.deepcopy(local_weights_delay[0]), local_delay_ew)# 第二个参数实际上是上一轮的pre_weights[1]
            global_update = Mean(weight_updates)
            # 重新恢复dict
            print("global update type is", type(global_update))
            global_weights = restoreWeight(std_dict, std_keys, global_update) 
        else:
            global_weights = Sag(epoch, average_weights(local_weights_delay[0]), len(local_weights_delay[0]),
                                                 local_delay_ew, copy.deepcopy(global_weights))

        # Update global weights
        pre_global_model.load_state_dict(global_model.state_dict())
        global_model.load_state_dict(global_weights)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            if c in attack_users and args.inverse_poison == True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False,  inverse_poison = True, idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, inverse_poison = False, idx=c)

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(train_test_acc)

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')

            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss, _ = test_inference(args, global_model, test_dataset)
        final_test_acc.append(test_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        # Schedular Update
        # for l in range(args.num_users):
        #     scheduler[l] = (scheduler[l] - 1) * ((scheduler[l] - 1) > 0)
        for l in all_users:
            if(scheduler[l] > 0):
                scheduler[l] = (scheduler[l] - 1)   


    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))

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

    file_n = f'accuracy_{args.update_rule}__{args.dataset}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.frac}_{args.seed}_{args.lam}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()










