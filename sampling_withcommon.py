import numpy as np
from torchvision import datasets, transforms
import copy




# Split the entire data into public data and users' data

def mnist_noniidcmm(dataset, num_users, num_commondata, alpha):

    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(1, num_users+1)}

    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))

    #Exclude the public data from local device
    all_idxs = list(set(all_idxs) - dict_users[0])
    total_data = len(all_idxs)
    
    dict_common = dict_users[0]
    idxs_labels = list(set(all_idxs) - dict_users[0])
    train_labels = dataset.train_labels.numpy()
    dict_users = dirichlet_split_noniid(len(dataset.classes),train_labels, alpha, num_users)

   

    return dict_users, dict_common


# def mnist_noniidcmm(dataset, num_users, num_commondata, alpha):

    
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     dict_users = {i: np.array([]) for i in range(1, num_users+1)}

#     dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))

#     #Exclude the public data from local device
#     all_idxs = list(set(all_idxs) - dict_users[0])
#     total_data = len(all_idxs)
#     # num_shards, num_imgs = 800, total_data//800 # 问题可能出在这
#     num_shards, num_imgs = num_users*2, total_data//(num_users*2) # 问题可能出在这
#     '''
#     #include the public data
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     '''


#     idx_shard = [i for i in range(num_shards)]
#     labels = dataset.train_labels.numpy()

#     idxs_labels = np.vstack((all_idxs, labels[all_idxs]))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]


#     idxs = idxs_labels[0,:]

#     for i in range(1,num_users+1):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard)- rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

#     dict_common = dict_users[0]
#     for i in range(num_users):
#         dict_users[i] = dict_users[i+1]
#     del dict_users[num_users]


#     return dict_users, dict_common


def dirichlet_split_noniid(n_classes, train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    # n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例 
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def cifar_noniidcmm(dataset, num_users, num_commondata):

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users+1)}
    print("type dict users[0]", type(dict_users[0]))
    idxs = np.arange(num_shards * num_imgs)
   

    # Exclude the public data from local device
    idxs = list(set(idxs))
    total_data = len(idxs)
    num_shards, num_imgs = 200, total_data//200

    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # common data
    idx_set = set(range(101))
    for rand in idx_set:
        dict_users[0] = np.concatenate((dict_users[0], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    print("type dict users[0]", type(dict_users[0]))
    print("type dict idxs", type(idxs))
    idxs = list(set(idxs) - set(dict_users[0]))
    dict_common = copy.deepcopy(dict_users[0])
    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)

    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]


    # for i in range(1,num_users+1):
    #     rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    # dict_common = dict_users[0]

    # for i in range(num_users):
    #     dict_users[i] = dict_users[i + 1]
    # del dict_users[num_users]
    train_labels = labels
    alpha = 1.0
    dict_users = dirichlet_split_noniid(len(dataset.classes),train_labels, alpha, num_users)


    return dict_users, dict_common










if __name__ == '__main__':
    if __name__ == '__main__':
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
