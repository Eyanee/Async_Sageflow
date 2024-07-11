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

from otherGroupingMethod import flatten_parameters
from visualdl import LogWriter

from update import LocalUpdate, test_inference, DatasetSplit
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from utils1 import *
from resnet import *
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

    exp_details(args)
    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    train_dataset, test_dataset, (user_groups, dict_common) = get_dataset(args)
    # writer = LogWriter(logdir="./log/histogram_test/sync_res_noattack_3")

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
    # print(global_model)
       # 加载参数  

    # params = torch.load('./global_model_parameters.pth')  

    # 使用加载的参数更新模型  

    # global_model.load_state_dict(params)

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    final_test_acc = []
    print_every = 2
    val_loss_pre, counter = 0, 0

    m = args.num_users
    all_users = np.arange(args.num_users)
    
    start = 20
    n = int(m * args.attack_ratio)
    attack_users = np.random.choice(range(start, m), n, replace=False)
    print("attack user is ",attack_users)
    





    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        

        loss_on_public = []
        entropy_on_public = []
        global_weights_rep = copy.deepcopy(global_model.state_dict())

        count = 0
        for idx in all_users:

            if idx in attack_users and args.data_poison==True:
                # print("label attack")
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], data_poison=True,idx=idx)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], data_poison=False, idx=idx)

            w, loss,_ = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )

            if idx in attack_users and args.model_poison ==True and epoch >=0:
                w = sign_attack(w, args.model_poison_scale)

            #### save for print test ####
            # count =0 
            # if epoch == 15:
            #     test_model = copy.deepcopy(global_mo


            if args.update_rule == 'Sageflow':
                # Averaging local weights via entropy-based filtering and loss-wegithed averaging
                global_weights,_ = Eflow(local_weights, loss_on_public,entropy_on_public,epoch)

            else:
                w_semi = copy.deepcopy(global_model.state_dict())
                for key in w_semi.keys():
                    if args.dataset =='cifar':
                        alpha = 0.05
                    elif args.dataset =='fmnist':
                        alpha = 0.1

                    elif args.dataset =='mnist':
                        alpha = 0.

                    w_semi[key] = w_semi[key] * (1 - alpha) + w[key] * (alpha)

            global_model.load_state_dict(w_semi)
            test_acc, test_loss , test_entropy = test_inference(args, copy.deepcopy(global_model), test_dataset)
            if idx in attack_users:
                print("|----epoch{} user{} type:malicious  Training Accuracy : {:.2f}%".format(epoch, idx, 100 * test_acc))
            else:
                print("|----epoch{} user{} type:benign  Training Accuracy : {:.2f}%".format(epoch, idx, 100 * test_acc))
                
            # print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))


        # Update global weights

        test_acc, test_loss , test_entropy = test_inference(args, copy.deepcopy(global_model), test_dataset)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
        final_test_acc.append(test_acc)

    print(f' \n Results after {args.epochs} global rounds of training:')



    # Final test accuarcy for global test dataset.
    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))

    exp_details(args)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    torch.save(global_model.state_dict(), './fmnist_AFL_model_parameters.pth')
    if args.data_poison == True:
        attack_type = 'data'
    elif args.model_poison == True:
        attack_type = 'model'
        model_scale = '_scale_' + str(args.model_poison_scale)
        attack_type += model_scale
    else:
        attack_type = 'no_attack'

    file_n = f'accuracy_sync_{args.update_rule}_{args.dataset}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.seed}_{args.Eflow}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()











