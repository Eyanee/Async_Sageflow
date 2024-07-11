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
from poison_optimization_test import Outline_Poisoning, add_small_perturbation,cal_ref_distance,model_dist_norm,cal_similarity
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
    # 加载参数  
    params = torch.load('./fmnist_AFL_model_parameters.pth')  

    # 使用加载的参数更新模型  

    global_model.load_state_dict(params)


    
    global_model.train()
    
    fi_global_model = copy.deepcopy(global_model)
    pre_global_model = copy.deepcopy(global_model)
    primitive_malicious = copy.deepcopy(global_model.state_dict())

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
    
    t = int(math.ceil(len(all_users)/args.staleness))
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
    # print("attack_user is", attack_users)
    # 为恶意的用户赋目标值
    start_idx = int(0.2* len(all_users))
    attack_users = np.random.choice(range(start_idx, len(all_users)), m, replace=False)
    print("attack user is ",attack_users)

    # for l in attack_users:
    #     clientStaleness[l] = TARGET_STALENESS   
        
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

        global_weights_rep = copy.deepcopy(global_model.state_dict())
        

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
            # gd_test = compute_gradient(w,global_model.state_dict(),std_keys,args.lr)

            ensure_1 += 1 # 平均分布
            if  idx in attack_users and args.new_poison == True and epoch >=0:
                print("sign attack scale is ",args.model_poison_scale)
                mal_dict = sign_attack(w, args.model_poison_scale)

                test_model = copy.deepcopy(global_model)
                test_model.load_state_dict(mal_dict)

                

                common_acc, common_loss_sync, common_entropy_sample = test_inference(args, copy.deepcopy(test_model),
                                                                                    DatasetSplit(train_dataset,
                                                                                        dict_common))
                local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(mal_dict))
                local_index_delay[ scheduler[idx] - 1 ].append(idx)
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
                loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
                entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)


               

                
            
        
                        

        for i in range(args.staleness):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    # Averaging delayed local weights via entropy-based filtering and loss-wegithed averaging
                    if len(local_weights_delay[i]) > 0:
                        w_avg_delay, len_delay = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                        
                        
                        test_model.load_state_dict(w_avg_delay)
                        mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                                    DatasetSplit(train_dataset,
                                                                                                        dict_common))
                        print("benign_acc is", mal_acc)
                        print("benign_loss is", mal_loss_sync)
                        print("benign_entropy is", mal_entropy_sample)
                        
                        pre_weights[i].append([w_avg_delay, len_delay])
                        # print("num1 of attacker is ",num_attacker_1)
                elif args.update_rule == 'AFL':
                    if len(local_weights_delay[i]) > 0:
                        # for idx, item in enumerate(local_weights_delay[i]):
                        pre_weights[i].append(local_weights_delay[i])
                        pre_indexes[i].append(local_index_delay[i])
                
                        
        if args.update_rule == 'Sageflow':
            sync_weights, len_sync = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)
            # Staleness-aware grouping
            
            test_model.load_state_dict(sync_weights)
            mal_acc, mal_loss_sync, mal_entropy_sample = test_inference(args, test_model,
                                                                                        DatasetSplit(train_dataset,
                                                                                            dict_common))
            print("sync_acc is", mal_acc)
            print("sync_loss is", mal_loss_sync)
            print("sync_entropy is", mal_entropy_sample)

            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))
        elif args.update_rule == 'AFL':
            update_params = copy.deepcopy(local_weights_delay[0])
            update_indexes = copy.deepcopy(local_index_delay[0])
            for item in local_delay_ew:
                update_params.extend(item)
            for item in local_index_ew:
                update_indexes.extend(item)
            
            w_semi = copy.deepcopy(global_model.state_dict())
            for key in w_semi.keys():
                if args.dataset =='cifar':
                    alpha = 0.05
                elif args.dataset =='fmnist':
                    alpha = 0.1

                elif args.dataset =='mnist':
                    alpha = 0.


            for idx, item in enumerate(update_params):
                for key in w_semi.keys():
                    w_semi[key] = w_semi[key] * (1 - alpha) + w[key] * (alpha)
                global_model.load_state_dict(w_semi)
                test_acc, test_loss , test_entropy = test_inference(args, copy.deepcopy(global_model), test_dataset)
                if idx in attack_users:
                    print("|----epoch{} user{} type:malicious  Training Accuracy : {:.2f}%".format(epoch, update_indexes[idx], 100 * test_acc))
                else:
                    print("|----epoch{} user{} type:benign  Training Accuracy : {:.2f}%".format(epoch, update_indexes[idx], 100 * test_acc))
        

            

       
        test_acc, test_loss, _ = test_inference(args, copy.deepcopy(global_model), test_dataset)
        final_test_acc.append(test_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        # Schedular Update
        for l in all_users:
            if(scheduler[l] > 0):
                scheduler[l] = (scheduler[l] - 1)   
                
        # Mal_parameters_list Update
        

    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))


    exp_details(args)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # torch.save(global_model.state_dict(), './mnist_iid_model_parameters.pth')

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










