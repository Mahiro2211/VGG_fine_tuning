#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import glob
import os.path as osp
import random
import    numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


# In[2]:



# In[3]:


# 生成种子保证实验的可重复性
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# In[4]:


#调用GPU
def try_gpu(i=0):
    if torch.cuda.device_count() > i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# In[5]:


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)), 
                transforms.RandomHorizontalFlip(),  # 图像增强
                transforms.ToTensor(),  
                transforms.Normalize(mean, std) 
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  
                transforms.CenterCrop(resize), 
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# In[6]:


os.chdir('../Downloads/pytorch_advanced-master/1_image_classification/')
def make_datapath_list(phase="train"):

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = [] 


    for path in glob.glob(target_path):  # 通配符抓取所有图片
        path_list.append(path)

    return path_list

train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

train_list


# In[7]:


size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class HymenopteraDataset(data.Dataset):


    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform 
        self.phase = phase  

    def __len__(self):

        return len(self.file_list)

    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = Image.open(img_path) 


        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])


        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]


        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label



train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')


index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])


# In[8]:


batch_size = 8


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)


dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


batch_iterator = iter(dataloaders_dict["train"]) 
inputs, labels = next(
    batch_iterator)  
print(inputs.size())
print(labels)


# In[9]:


use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(4096 , 2)
net.train()


# In[10]:


criterion = nn.CrossEntropyLoss()

params_to_update = []

update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

print("-----------")
print(params_to_update)


# In[11]:


optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
net = net.to(device=try_gpu())
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device=try_gpu())


# In[12]:


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0 
            epoch_corrects = 0 


            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device=try_gpu())
                lables = labels.to(device=try_gpu())
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1) 

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)  

                    epoch_corrects += torch.sum(preds == labels.data)


            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


# In[13]:


dataloaders_dict


# In[14]:


num_epochs=2
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# # 微调(fine-tune)

# In[22]:


train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

batch_size = 8

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


# In[23]:


use_pretrained = True  
net = models.vgg16(pretrained=use_pretrained)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
net.train()
criterion = nn.CrossEntropyLoss()


# In[24]:


# 微调中需要学习的参数保存到下面三个列表中

params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 需要学习的网络层
update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# 打开对应网络层的梯度计算
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1保存：", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2保存：", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3保存：", name)

    else:
        param.requires_grad = False
        print("不进行计算梯度，我们不学习这层", name)


# In[25]:


optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)


# In[26]:


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用的设备是：", device)

   
    net.to(device)

    
    torch.backends.cudnn.benchmark = True


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  
            else:
                net.eval()   

            epoch_loss = 0.0  
            epoch_corrects = 0 

         
            if (epoch == 0) and (phase == 'train'):
                continue

        
            for inputs, labels in tqdm(dataloaders_dict[phase]):

          
                inputs = inputs.to(device)
                labels = labels.to(device)

               
                optimizer.zero_grad()

               
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  
                    _, preds = torch.max(outputs, 1)

  
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

      
                    epoch_loss += loss.item() * inputs.size(0)  
  
                    epoch_corrects += torch.sum(preds == labels.data)

     
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))




