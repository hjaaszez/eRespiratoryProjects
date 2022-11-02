'''
pytorchによるDNNモデルの学習
'''
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from dataloader import RespireSoundDataset
from model.CRNN import Model

class TrainDNN:
    def __init__(self, device, dataloader, optimizer, criterion, num_epochs, model, save_folder, model_name):
        self.device      = device
        self.dataloader  = dataloader
        self.optimizer   = optimizer
        self.criterion   = criterion
        self.num_epochs  = num_epochs
        self.model       = model
        self.save_folder = save_folder
        self.model_name  = model_name

        self.best_accuracy = 0
    
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            print(f'best accuracy is {self.best_accuracy}')
            predict_list = []
            label_list   = []

            for phase in ['train', 'eval']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_sum=0
                    corrects=0
                    total=0

                    with tqdm(total = len(self.dataloader[phase]), unit='batch') as pbar:
                        pbar.set_description(f'Epoch[{epoch}/{self.num_epochs}]({phase})')

                        for spectrogram, labels in self.dataloader[phase]:   
                            spectrogram, labels = spectrogram.to(self.device), labels.to(self.device)
                            
                            output = self.model(spectrogram)

                            loss   = self.criterion(output, labels)

                            if phase == 'train':
                                self.optimizer.zero_grad()
                                loss.backward()
                                self.optimizer.step()

                            predict   = torch.argmax(output, dim=1) 
                            corrects += (predict == labels).sum()
                            total    += spectrogram.size(0)
                            loss_sum += loss * spectrogram.size(0) 

                            accuracy     = corrects.item() / total
                            running_loss = loss_sum / total
                            pbar.set_postfix({"loss": running_loss.item(),"accuracy": accuracy })
                            pbar.update(1)

                            if phase == 'eval':
                                predict_list = predict_list + predict.tolist()
                                label_list   = label_list   + labels.tolist()

            if accuracy_score(label_list, predict_list) >= self.best_accuracy:
                print('best_accuracy is update!')
                self.save_model()
                self.best_accuracy = accuracy_score(label_list, predict_list)
        
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_folder}/{self.model_name}.pth')


if __name__ == '__main__':
     
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:1', help='set gpu device')
    parser.add_argument('--batch_size', type=int, default=256, help='set batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='set num_epochs')
    parser.add_argument('--feats_data_path', type=str, default='./data/feats', help='set save feats data path')
    parser.add_argument('--eval_dataset_id', type=int, help='set eval dataset_id')
    parser.add_argument('--model_name', type=str, default='model', help='set model name')
    args = parser.parse_args()

    device          = torch.device(args.device)
    batch_size      = args.batch_size
    num_epochs      = args.num_epochs
    data_root       = args.feats_data_path
    eval_dataset_id = args.eval_dataset_id
    model_name      = args.model_name
    
    save_folder = f'./assets/dataset_{eval_dataset_id}/model'
    os.makedirs(save_folder, exist_ok = True)
    num_classes = 4

    model     = Model().to(device)
    weights   = torch.tensor([1.0, 2.0, 4.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, amsgrad=True)

    transform = {
        'train': transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
        ]),
        'eval': transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
        ])
    }

    dataset = {
        'train' : RespireSoundDataset(data_root = data_root, 
                                      eval_dataset_id = eval_dataset_id, 
                                      transform = transform['train'],
                                      trainable = True),

        'eval'  : RespireSoundDataset(data_root = data_root, 
                                      eval_dataset_id = eval_dataset_id, 
                                      transform = transform['eval'],
                                      trainable = False)
    }

    dataloader = {
        'train' : DataLoader(dataset['train'], batch_size = batch_size, shuffle = True, num_workers = 2),
        'eval'  : DataLoader(dataset['eval'], batch_size = batch_size, shuffle = False, num_workers = 2)
    }

    trainer = TrainDNN(device     = device,
                       dataloader = dataloader,
                       optimizer  = optimizer,
                       criterion  = criterion,
                       num_epochs = num_epochs,
                       model      = model,
                       save_folder= save_folder,
                       model_name = model_name)
    trainer.train()


