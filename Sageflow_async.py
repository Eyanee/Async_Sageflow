#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
# from visualdl import LogWriter

import torch

from update import LocalUpdate, test_inference, DatasetSplit
from poison_optimization_cifar import Outline_Poisoning, add_small_perturbation,cal_ref_distance,model_dist_norm,cal_similarity
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from resnet import *
from utils1 import *
from added_funcs import poison_Mean
import csv
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import os
from otherGroupingMethod import *
from otherPoisoningMethod import *


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

    # writer = LogWriter(logdir="./log/histogram_test/async_res_noattack_3")

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
    # # 加载参数  

    # params = torch.load('./cifar_LDR_200_noniid_model_parameters_1.pth')  
    params = torch.load('./global_model_parameters.pth')  

    # 使用加载的参数更新模型  

    global_model.load_state_dict(params)
    
    global_model.train()
    

    global_weights = global_model.state_dict()

    train_accuracy = []
    final_test_acc = []
    print_every = 1

    pre_weights = {} 
    pre_indexes = {}
    pre_grad = {}
    pre_loss = {}
    # 多级staleness的存储
    for i in range(args.staleness + 1): 
        if i != 0:
            pre_weights[i] = []
            pre_indexes[i] = []
            pre_grad[i] = []
            pre_loss[i] = []

    # Device schedular
    scheduler = {}
    
    # Staleness 设定
    clientStaleness = {}
    
    # 目标Staleness设定
    TARGET_STALENESS  = 1

    # 投毒状态标志
    poisoned = False
    # 其他参数
    distance_ratio = 1
    adaptive_accuracy_threshold = 0.8 ## 需要调整
    pinned_accuracy_threshold = 0.
     ## 需要调整

 
    for l in range(args.num_users):
        scheduler[l] = 0
        clientStaleness[l] = 0

    global_epoch = 0
    
    all_users = np.arange(args.num_users)
    m = int(args.num_users * args.attack_ratio)
    n = args.num_users - m
    attack_users = all_users[-m:]
    print("attack user num is ",m)
    
    t = int(math.ceil(n/args.staleness))
    print(" t is ",t )
    
    # 为正常用户赋固定的Staleness值
    for i in range(args.staleness):
        if i == args.staleness +1:
            front_idx = int(t * i)
            end_idx = n-  1
        else:
            front_idx = int(t * i)
            end_idx = front_idx + t
        for j in range(front_idx, end_idx):
            clientStaleness[j] = i + 1 
    print("attack_user is", attack_users)
    
    # 为恶意用户赋目标的Staleness值
    # last_index = n
    # for l in attack_users:
    #     # clientStaleness[l] = TARGET_STALENESS  
    #     ti = int(math.ceil(m/(args.staleness-1)))
    #     print("ti is", ti)
    #     # 为恶意用户赋固定的Staleness值
    #     for i in range(args.staleness-1):
    #         if i == args.staleness :
    #             front_idx = int(ti * i) +last_index
    #             end_idx = m+n-  1
    #         else:
    #             front_idx = int(ti * i) + last_index
    #             end_idx = front_idx + ti
    #         for j in range(front_idx, end_idx):
    #             clientStaleness[j] = i + 2

    for l in attack_users:
        clientStaleness[l] = TARGET_STALENESS   
        
    # 恶意用户的历史存储
    MAX_STALENESS = args.staleness
    mal_parameters_list = {}
    for i in range(MAX_STALENESS):
        mal_parameters_list[i] = {}
    
    mal_grad_list = []

    
    
    std_keys = get_key_list(global_model.state_dict().keys())

    mal_rand = global_model.state_dict()
    
            
    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        loss_on_public = {}
        entropy_on_public = {}
        local_index_delay = {}
        local_grad_delay = {}
        malicious_models = []
        mal_grad_list = []

        for i in range(args.staleness + 1):
            loss_on_public[i] = []
            entropy_on_public[i] = []
            local_weights_delay[i] = []
            local_index_delay[i] = []
            local_grad_delay[i] = []


        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()
        

        # After round, each staleness group is adjusted
        local_delay_ew = copy.deepcopy(pre_weights[1])
        local_index_ew = copy.deepcopy(pre_indexes[1])
        local_delay_gd = copy.deepcopy(pre_grad[1])
        local_delay_loss = copy.deepcopy(pre_loss[1])

        for i in range(args.staleness):
            if i != 0:
                pre_weights[i] = copy.deepcopy(pre_weights[i+1])
                pre_indexes[i] = copy.deepcopy(pre_indexes[i+1])
                pre_grad[i] = copy.deepcopy(pre_grad[i + 1])
                pre_loss[i] = copy.deepcopy(pre_loss[i + 1])

        pre_weights[args.staleness] = [] # 对staleness的权重
        pre_indexes[args.staleness] = []
        pre_grad[args.staleness] = []
        pre_loss[args.staleness] = []

        ensure_1 = 0
        count = 0 
        for idx in all_users:

            if scheduler[idx] == 0: # 当前轮次提交
                if idx in attack_users and args.data_poison == True: # 原 data_posion 入口
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=True)

                else:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=False)

                # Ensure each staleness group has at least one element
                scheduler[idx] = clientStaleness[idx]  # 重新赋值staleness
                print("current submit client idx and staleness is ", idx ,scheduler[idx])

            else:

                continue

            w, loss,_ = local_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )

            check_distance = model_dist_norm(w,copy.deepcopy(global_model.state_dict()))
            print("check distance is ", check_distance)
            test_simliarity = cal_similarity( global_model.state_dict(),w)
            print("benign similarity is ", test_simliarity)

            gd_test = compute_gradient(w,global_model.state_dict(),std_keys,args.lr)
            
            ensure_1 += 1 # 平均分布
            if idx in attack_users and args.model_poison == True and epoch >=6 - MAX_STALENESS:
                # if args.poison_methods == 'ourpoisonMethod':
                #     print("here")
                mal_parameters_list[0][idx] = w # 加入malicious_list
                mal_grad_list.append(gd_test)

            elif  idx in attack_users and args.new_poison == True and epoch >=0:
                print("sign attack scale is ",args.model_poison_scale)
                mal_dict = sign_attack(w, args.model_poison_scale)

                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(mal_dict)

                mal_grad = compute_gradient(mal_dict,global_model.state_dict(),std_keys,args.lr)

                

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))
                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(mal_dict))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_grad))
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
                # print(" sign acc is ", common_acc)
                # print(" sign loss is ", common_loss_sync)
                # print(" sign entropy is ",common_entropy_sample )
                # benign_distance = model_dist_norm(mal_dict,copy.deepcopy(global_model.state_dict()))
                # print("sign distance is ", benign_distance)
            else:     
                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(w)

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))
                # common_grad = compute_gradient(w,global_model.state_dict(),std_keys,args.lr)

                # common_params =restoregradients(copy.deepcopy(global_model.state_dict()), std_keys, args.lr * common_grad)

                # test_distance= model_dist_norm(common_params,w)
                # print("test_distance is",test_distance)


                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(w))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
                local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(gd_test))
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)
                print(" benign loss is ", common_loss_sync)
                print(" benign acc is ", common_acc)
                


                # norm_1 = torch.norm(gd_test, p =2 )
                # print("norm1 res is ", norm_1)

                
            
        if args.model_poison == True and epoch >=0 :
            if args.poison_methods == 'ourpoisonMethod':
                # malicious_models = list(mal_parameters_list[MAX_STALENESS - 1].values()) #本地模拟的陈旧度上限
                malicious_models = list(mal_parameters_list[0].values())
                # local_dict = mal_parameters_list[0][attack_users[0]]s
                # previous_dict = mal_parameters_list[1][attack_users[0]]
                
                pinned_accuracy_threshold = 0.5 # 
                adaptive_accuracy_threshold = pinned_accuracy_threshold
                if epoch ==  0:
                    distance_threshold = cal_ref_distance(malicious_models,copy.deepcopy(global_model), 0.8) 
                    mal_rand = add_small_perturbation(global_model, args, pinned_accuracy_threshold, train_dataset, distance_threshold,perturbation_range=(-0.1, 0.1))
                    test_model.load_state_dict(mal_rand)
                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))   
                    print("mal_rand_acc is", mal_acc)
                    print("mal_rand loss is", mal_loss_sync)


                distance_threshold = cal_ref_distance(malicious_models,copy.deepcopy(global_model), 0.8) 
                print("distance Threshold  is ",distance_threshold)
                malicious_dict = Outline_Poisoning(args, copy.deepcopy(global_model), malicious_models, 
                                                                train_dataset, distance_threshold, pinned_accuracy_threshold,copy.deepcopy(mal_rand))
                
            
                test_model.load_state_dict(malicious_dict)
                mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                            DatasetSplit(train_dataset,
                                                                                                        dict_common))
                
                mal_grad = compute_gradient(malicious_dict,global_model.state_dict(),std_keys,args.lr)
                print("mal_acc is", mal_acc)
                print("mal loss is", mal_loss_sync)
                norm_2 = torch.norm(mal_grad, p =2 )
                print("mal norm is ", norm_2)
                mal_distance = model_dist_norm(malicious_dict,copy.deepcopy(global_model.state_dict()))
                print("mal distance is ", mal_distance)

                for idx in attack_users:
                        
                    local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(malicious_dict))
                    local_index_delay[ scheduler[idx] - 1 ].append(idx)
                    loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                    entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                    #
                    local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_grad))
        
            elif args.poison_methods == 'LA':
                    malicious_dicts= LA_attack(args, mal_grad_list, m,copy.deepcopy(global_model),std_keys)
                    global_weights = copy.deepcopy(global_model.state_dict())
                    test_model.load_state_dict(global_weights)
                    optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                    for mal_dict in malicious_dicts:
                        

                        model_grads=[]
                        start_idx = 0
                        for i, param in enumerate(global_model.parameters()):
                            param_=mal_dict[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                            start_idx=start_idx+len(param.data.view(-1))
                            param_=param_.to(device)
                            model_grads.append(param_)

                        optimizer_fed.step(model_grads)
                        # test_model.load_state_dict(malicious_dicts[num])
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                
                        # mal_grad = compute_gradient(malicious_dicts[num],global_model.state_dict(),std_keys,args.lr)
                        # print("mal_acc is", mal_acc)
                        # norm_2 = torch.norm(mal_grad, p =2 )
                        # print("mal norm is ", norm_2)
                        # mal_distance = model_dist_norm(malicious_dicts[num],copy.deepcopy(global_model.state_dict()))
                        # print("mal distance is ", mal_distance)
                        
                        local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(test_model.state_dict()))
                        local_index_delay[ scheduler[idx] - 1 ].append(idx)
                        loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                        entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                        #
                        local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(mal_dict))
                        test_model.load_state_dict(global_weights)  # 重新清零


            else:
                # for idx in attack_users  :
                if  scheduler[idx] == clientStaleness[idx]:
                    if args.poison_methods == 'LIE':
                        print("len of [max] is ", len(list(mal_parameters_list[MAX_STALENESS - 1].values())))
                        print("len of [0] is ", len(list(mal_parameters_list[0].values())))
                        malicious_dict = LIE_attack(list(mal_parameters_list[0].values()))
                        malicious_grads=malicious_dict
                    
                    elif args.poison_methods == 'min_max':
                        print("len of [0] is ", len(list(mal_parameters_list[0].values())))
                        malicious_dict = min_max(args, list(mal_parameters_list[0].values()))
                    elif args.poison_methods == 'min_sum':
                        test_model.load_state_dict(global_weights)
                        optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                        malicious_grads = min_sum(args, mal_grad_list)
                        model_grads=[]
                        start_idx = 0
                        for i, param in enumerate(global_model.parameters()):
                            param_=malicious_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                            start_idx=start_idx+len(param.data.view(-1))
                            param_=param_.to(device)
                            model_grads.append(param_)

                        optimizer_fed.step(model_grads)
                    elif args.poison_methods == 'Grad':
                        # benign_grads  = list()
                        # for params in  list(mal_parameters_list[0].values()):
                        #     grad  = compute_gradient(params,global_model.state_dict(),std_keys,args.lr)
                        #     benign_grads.append(grad)
                            
                        # benign_grads =  torch.stack(benign_params)
                        test_model.load_state_dict(global_weights)
                        optimizer_fed = Adam(test_model.parameters(), lr=0.01)
                        malicious_grads = Grad_median(args, mal_grad_list, m)
                        model_grads=[]
                        start_idx = 0
                        for i, param in enumerate(global_model.parameters()):
                            param_=malicious_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
                            start_idx=start_idx+len(param.data.view(-1))
                            param_=param_.to(device)
                            model_grads.append(param_)

                        optimizer_fed.step(model_grads)

                        
                        # malicious_dict = restoreWeight(copy.deepcopy(global_model.state_dict()), std_keys, args.lr * malicious_dict)
                    

                    # test_model.load_state_dict(malicious_dict)
                    mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                                DatasetSplit(train_dataset,
                                                                                                    dict_common))
                    malicious_dict = copy.deepcopy(test_model.state_dict())
                    # mal_grad = compute_gradient(malicious_dict,global_model.state_dict(),std_keys,args.lr)
                    print("mal_acc is", mal_acc)
                    print("mal_loss is", mal_loss_sync)
                    print("mal_entropy is", mal_entropy_sample)
                    for idx in attack_users  :
                        local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(malicious_dict))
                        local_index_delay[ scheduler[idx] - 1 ].append(idx)
                        loss_on_public[scheduler[idx] - 1].append(mal_loss_sync)
                        entropy_on_public[scheduler[idx] - 1].append(mal_entropy_sample)
                
                        local_grad_delay[scheduler[idx] - 1].append(copy.deepcopy(malicious_grads))
                    
                        

        for i in range(args.staleness):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    # Averaging delayed local weights via entropy-based filtering and loss-wegithed averaging
                    if len(local_weights_delay[i]) > 0:
                        w_avg_delay, len_delay = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                        
                        
                        test_model.load_state_dict(w_avg_delay)
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                        print("benign_acc is", mal_acc)
                        print("benign_loss is", mal_loss_sync)
                        print("benign_entropy is", mal_entropy_sample)
                        
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        # print("num1 of attacker is ",num_attacker_1)
                elif args.update_rule == 'AFA':
                    if len(local_weights_delay[i]) > 0:
                        std_keys = get_key_list(global_model.state_dict().keys())
                        w_avg_delay, len_delay = pre_AFA(copy.deepcopy(global_model.state_dict()), std_keys, local_weights_delay[i], local_index_delay[i], device)
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                elif args.update_rule == 'Median':
                    if len(local_weights_delay[i]) > 0:
                        # std_keys = get_key_list(global_model.state_dict().keys())
                        # w_avg_delay, len_delay = pre_Median(copy.deepcopy(global_model.state_dict()), std_keys, local_weights_delay[i])
                        # pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        pre_weights[i].append(local_weights_delay[i])

                elif args.update_rule == 'Trimmed_mean':
                    if len(local_weights_delay[i]) > 0:
                        # std_keys = get_key_list(global_model.state_dict().keys())
                        # w_avg_delay, len_delay = pre_Trimmed_mean(copy.deepcopy(global_model.state_dict()), std_keys, local_weights_delay[i])
                        # pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'norm_bounding':
                    if len(local_weights_delay[i]) > 0:
                        # 暂时还没实现
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'Zeno':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i] = np.concatenate((pre_weights[i], local_weights_delay[i]), axis=0)
                        # pre_grad[i] = pre_grad[i] + local_grad_delay[i]
                        # pre_loss[i] = np.concatenate((pre_loss[i], loss_on_public[i]), axis=0)
                elif args.update_rule == 'AFLGuard':
                    if len(local_weights_delay[i]) > 0:
                        # for j in range(len(local_weights_delay[i])):
                        #     # pre_weights[i].append([local_weights_delay[i][j], local_grad_delay[i][j]])
                        pre_weights[i].append(local_weights_delay[i])
                elif args.update_rule == 'Zenoplusplus':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                        pre_indexes[i].append(local_index_delay[i])
                        pre_grad[i].append(local_grad_delay[i])
                elif args.update_rule == 'FLARE':
                    if len(local_weights_delay[i]) > 0:
                        pre_weights[i].append(local_weights_delay[i])
                else:
                    # if len(local_weights_delay[i]) > 0:F
                    #     pre_weights[i].append(local_weights_delay[i])
                    if len(local_weights_delay[i]) > 0:
                        w_avg_delay = average_weights(local_weights_delay[i])
                        len_delay = len(local_weights_delay[i])
                        pre_weights[i].append({epoch: [w_avg_delay, len_delay]})
                        
                        if i ==1:
                            test_model.load_state_dict(w_avg_delay)
                            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                                        DatasetSplit(train_dataset,
                                                                                                            dict_common))
                            print("benign_acc is", mal_acc)
                            print("benign_loss is", mal_loss_sync)
                            print("benign_entropy is", mal_entropy_sample)
                
                        
        if args.update_rule == 'Sageflow':
            sync_weights, len_sync = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)
            # Staleness-aware grouping
            
            test_model.load_state_dict(sync_weights)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))
            print("sync_acc is", mal_acc)
            print("sync_loss is", mal_loss_sync)
            print("sync_entropy is", mal_entropy_sample)

            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
        elif args.update_rule == 'AFA':
            std_keys = get_key_list(global_model.state_dict().keys())
            sync_weights ,len_sync = pre_AFA(copy.deepcopy(global_model.state_dict()), std_keys, local_weights_delay[0], local_index_delay[0], device)
            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
        elif args.update_rule == 'Median':
            std_keys = get_key_list(global_model.state_dict().keys())
            current_param = []
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            sync_weights, len_sync = pre_Median(copy.deepcopy(global_model.state_dict()), std_keys, current_param)
            global_weights = update_weights(global_model.state_dict(),sync_weights)
            # global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
            #                                          copy.deepcopy(global_weights))
            # test_model.load_state_dict(global_weights)
            # avg_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
            #                                                                             test_dataset)
            # print("median_weights acc is ", avg_acc)
        

        elif args.update_rule == 'Trimmed_mean': # ?有问题需要重新考虑
            std_keys = get_key_list(global_model.state_dict().keys())
            current_param = []
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            sync_weights, len_sync = pre_Trimmed_mean(copy.deepcopy(global_model.state_dict()), std_keys, current_param)
            global_weights = update_weights(global_model.state_dict(),sync_weights)
            
        elif args.update_rule == 'norm_bounding':
            global_weights = norm_clipping(global_model, local_weights_delay[0],local_delay_ew,std_keys,  args.lr)
        elif args.update_rule == 'Zeno':
            print("Zeno")
            local_delay_ew = np.concatenate((local_delay_ew, local_weights_delay[0]), axis=0)
            # local_delay_gd = local_delay_gd + local_grad_delay[0]
            # local_delay_loss = np.concatenate((local_delay_loss, loss_on_public[0]), axis=0)
            global_weights = Zeno(local_delay_ew, local_delay_gd,  args,
                                    copy.deepcopy(global_model),
                                    DatasetSplit(train_dataset, dict_common), epoch)
        elif args.update_rule == 'AFLGuard':
            current_param = []
            global_test_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[0], idx=idx,
                                              data_poison=False)

            # for j in range(len(local_weights_delay[0])):
                # current_param.append([local_weights_delay[0][j], local_grad_delay[0][j]])
            current_param = copy.deepcopy(local_weights_delay[0])
            
            for k in local_delay_ew:
                current_param.extend(k)
            # print("len current param", len(current_param))
            global_weights = AFLGuard(current_param, global_model, global_test_model, epoch, std_keys, args.lr, lamda = 1.8)
        
        elif args.update_rule == 'Zenoplusplus':
            current_grad = copy.deepcopy(local_grad_delay[0])
            current_param = copy.deepcopy(local_weights_delay[0])
            current_index = copy.deepcopy(local_index_delay[0])
            global_test_model = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_common, idx=idx,
                                              data_poison=False)
            print("len current grad ",len(current_grad))
            for item in local_delay_gd:
                current_grad.extend(item)
            for item in local_delay_ew:
                current_param.extend(item)
            for item  in local_index_ew:
                current_index.extend(item)
            print("len all grad", len(current_grad))
            global_param_update = update_weights_zeno(args,copy.deepcopy(global_model),epoch,DatasetSplit(train_dataset,
                                                                                                        dict_common))
            # global_grad = compute_gradient(global_param_update,global_model.state_dict(),std_keys,args.lr)
            
            accept_list = Zenoplusplus(args, copy.deepcopy(global_model.state_dict()),current_param,global_param_update,std_keys, current_index)
            print("len accept list", len(accept_list) )
            if len(accept_list)== 0:
                global_weights = global_model.state_dict()
            else:
                global_weights , avg_weights= Fedavg(args, epoch, accept_list, global_model)

        elif args.update_rule == 'FLARE':
            update_params = copy.deepcopy(local_weights_delay[0])
            for item  in local_delay_ew:
                update_params.extend(item)
            global_weights = FLARE(args, global_model, update_params,DatasetSplit(train_dataset,dict_common) )
            

        else:
            w_avg = average_weights(local_weights_delay[0])
            print(len(local_weights_delay[0]))
            global_weights= Sag(epoch, w_avg, len(local_weights_delay[0]),
                                                 local_delay_ew, copy.deepcopy(global_weights))
            test_model.load_state_dict(w_avg)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                
            # Fedavg
            # all_weights = copy.deepcopy(local_weights_delay[0])
            # # all_weights.extend(local_delay_ew)
            # for item in local_delay_ew:
            #     all_weights.extend(item)
            # global_weights , avg_weights= Fedavg(args, epoch, all_weights, global_model)
            # test_model.load_state_dict(avg_weights)
            # avg_acc, mal_loss_sync, mal_entropy_sample, mal_grad = test_inference(args, test_model,
            #                                                                             test_dataset)
            print("w_avg acc is ", mal_acc)
            mal_distance = model_dist_norm(w_avg,copy.deepcopy(global_model.state_dict()))
            print("w_avg distance is ", mal_distance)

            

        # Update global weights
        # pre_global_model.load_state_dict(global_model.state_dict())
        global_model.load_state_dict(global_weights)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            if c in attack_users and args.inverse_poison == True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False,  idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, idx=c)

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(train_test_acc)

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')

            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss, _ = test_inference(args, copy.deepcopy(global_model), test_dataset)
        final_test_acc.append(test_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        # Schedular Update
        for l in all_users:
            if(scheduler[l] > 0):
                scheduler[l] = (scheduler[l] - 1)   
                
        # Mal_parameters_list Update
        
        if idx in attack_users and args.model_poison == True and epoch >=6 - MAX_STALENESS:
            for i in range(MAX_STALENESS-1 , 0 , -1):
                mal_parameters_list[i] = copy.copy(mal_parameters_list[i-1])
            mal_parameters_list[0] = {}
            


    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))

    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))


    exp_details(args)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # torch.save(global_model.state_dict(), './cifar_iid_200_model_parameters.pth')

    if args.data_poison == True:
        attack_type = 'data'
    elif args.model_poison == True:
        attack_type = 'model'
        model_scale = '_scale_' + str(args.model_poison_scale)
        attack_type += model_scale
    else:
        attack_type = 'no_attack'

    file_n = f'accuracy_{args.update_rule}__{args.poison_methods}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.frac}_{args.seed}_{args.lam}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()










