import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import copy
import math
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from customLossFuncs import CustomDistance1




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
    def __init__(self, args, dataset, idxs, idx,data_poison, labelmap = {}, delay=False):
        self.args = args
        self.idx= idx
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'

        self.criterion = nn.NLLLoss().to(self.device)
        self.delay = delay
        self.data_poison = data_poison
        self.labelmap = labelmap

    def train_val_test(self, dataset, idxs):

        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=max(int(len(idxs_test)/10),1), shuffle=False)

        return trainloader, testloader

# 修改这一部分的梯度更新方式
    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []
        epoch_grad = []

        if self.args.optimizer == 'sgd':

            lr = self.args.lr
            lr = lr * (0.5) ** (global_round // self.args.lrdecay)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        elif self.args.optimizer == 'adam':

            lr = self.args.lr
            lr = lr * (0.5) ** (global_round // self.args.lrdecay)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_grad = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.data_poison ==True:
                    labels = (labels+1)%10
                    # for idx,label in enumerate(labels):
                    #     if label == 7:
                    #         labels[idx] = 1
                    #     elif label == 1:
                    #         labels[idx] = 7
                    
                model.zero_grad()
                 # 修改点1：设置模型参数需要梯度
                for param in model.parameters():
                    param.requires_grad_(True)
                log_probs, _ ,PLR= model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                grad = torch.cat([p.grad.view(-1) for p in model.parameters()])
                # print(grad)
                batch_grad.append(grad)               

                batch_loss.append(loss.item())
            # print(batch_grad)
            # grad_tensor = torch.tensor(np.array([item.cpu().detach().numpy() for item in batch_grad])).cuda()
            # grad_mean = torch.mean(grad_tensor, dim=0)
            
            x = batch_grad[0]
            for i in range(1, len(batch_grad)):
                x += batch_grad[i]
            x = x / len(batch_grad)
            epoch_grad.append(x)
            # print(x)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        xx = epoch_grad[0]
        for i in range(1, len(epoch_grad)):
            xx += epoch_grad[i]
        xx = xx / len(epoch_grad)
        return_grad = xx
        
        # print(model.state_dict())
        # print(xx)
        
        return model.state_dict(),  sum(epoch_loss) / len(epoch_loss) , return_grad


    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs,_ ,PLR= model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):

    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    batch_losses = []
    batch_entropy = []
    batch_grad = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()


            output, out ,PLR = model(images)

            
            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            
            entropy  = -1.0 * Information.sum(dim=1) # size [64]
            average_entropy = entropy.mean().item()
            

            batch_loss = criterion(output, labels)
            batch_losses.append(batch_loss.item())

            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            pred_dec = torch.eq(pred_labels, labels)
            current_acc = torch.sum(pred_dec).item() + 1e-8

            batch_entropy.append(average_entropy)

            correct += current_acc
            total += len(labels)



    accuracy  = correct/total

        

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)


