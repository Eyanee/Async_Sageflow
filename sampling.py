

import numpy as np
from torchvision import datasets, transforms

# def mnist_iid(dataset, num_users):

#     num_items = int(len(dataset)/(num_users + 1))
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users + 1):

#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))


#         all_idxs = list(set(all_idxs) - dict_users[i])


#     return dict_users, dict_users[num_users]

import numpy as np

def mnist_iid(dataset, number_user):
    # 计算每个用户的数据量（均分）
    num_items_per_user = len(dataset) // (number_user+1)
    
    # 创建字典以存储每个用户的图像索引
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    # 为前 number_user 个用户均分数据
    for i in range(number_user+1):
        dict_users[i] = set(np.random.choice(all_idxs, num_items_per_user, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    # 处理最后一个用户，每个类别只选择2个图像
    # last_user_idx = number_user
    # for class_idx in range(10):  # 假设类别总数为10
    #     class_idxs = [idx for idx in range(class_idx * len(dataset)//10, (class_idx + 1) * len(dataset)//10)]
    #     np.random.shuffle(class_idxs)
        
    #     # 选择两个图像
    #     selected_idxs = class_idxs[:2]
        
    #     # 添加到最后一个用户的集合中
    #     dict_users[last_user_idx] = dict_users.get(last_user_idx, set()) | set(selected_idxs)
    
    return dict_users, dict_users[number_user]



def mnist_noniid(dataset, num_users):


    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]


    idxs = idxs_labels[0,:]



    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard)- rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



def mnist_noniid_unequal(dataset, num_users):


    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]



    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1, size= num_users)

    random_shard_size = np.around(random_shard_size / sum(random_shard_size)*num_shards)

    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):

            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        random_shard_size = random_shard_size -1


        for i in range(num_users):
            if len(idx_shard) ==0:
                continue
            shard_size = random_shard_size[i]


            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]

            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


        if len(idx_shard)>0:

            shard_size = len(idx_shard)


            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    num_shards, num_imgs = 200, 250 # 50000用作训练集
    idxs = np.arange(num_shards * num_imgs)
    total_data = len(idxs)
    num_shards, num_imgs = 200, total_data//200
    idx_shard = [i for i in range(num_shards)]
    dict_common = np.array([] ,dtype=np.int64)

    idxs = np.arange(num_shards * num_imgs)
    
   

    # Exclude the public data from local device
    idxs = list(set(idxs))
    total_data = len(idxs)
    b = []
    for i in idxs:
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs_ordered = idxs_labels[0,:]
    idx_set = set(range(10))
    num_shards = len(idxs_ordered)//10

    for rand in idx_set:
        dict_common = np.concatenate((dict_common, idxs_ordered[rand*num_shards:rand*num_shards+50]), axis=0)

    # for idx, item in enumerate(dict_common):  
    #     if idx <= 10:
    #         print(f"Item: {item}, Type: {type(item)}") 
    
    all_idxs = list(set(idxs) - set(dict_common))

    num_items = int(len(all_idxs)/(num_users))
    print("num_items is ", num_items)

    dict_users = {}
    for i in range((num_users)):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False ))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    # 处理最后一个用户，每个类别只选择2个图像
    # last_user_idx = num_users
    # for class_idx in range(10):  # 假设类别总数为10
    #     class_idxs = [idx for idx in range(class_idx * len(dataset)//10, (class_idx + 1) * len(dataset)//10)]
    #     np.random.shuffle(class_idxs)
        
    #     # 选择两个图像
    #     selected_idxs = class_idxs[:2]
        
    #     # 添加到最后一个用户的集合中
    #     dict_users[last_user_idx] = dict_users.get(last_user_idx, set()) | set(selected_idxs) 
        
        
    return dict_users, dict_common

def cifar_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    b = []
    for i in range(len(dataset)):
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]



    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))
    num = 100
    d = mnist_noniid(dataset_train, num)




















