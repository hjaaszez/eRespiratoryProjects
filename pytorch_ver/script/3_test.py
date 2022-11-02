'''
学習済みDNNモデルの推論
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
from evaluation import Evaluation_icbhi

class EvaluateDNN:
    def __init__(self, device, dataloader, model, save_folder, model_name):
        self.device      = device
        self.dataloader  = dataloader
        self.model       = model
        self.model.load_state_dict(torch.load(f'{save_folder}/{model_name}.pth'))
        self.save_folder = save_folder
        self.model_name  = model_name
    
    def evalate(self):
        
        predict_list = []
        label_list   = []
        self.model.eval()

        with torch.no_grad():
            
            for spectrogram, labels in self.dataloader['eval']:   
                spectrogram, labels = spectrogram.to(self.device), labels.to(self.device)
                
                output = self.model(spectrogram)
                # score算出のためone hot encoding
                labels = nn.functional.one_hot(labels, num_classes = 4)

                predict_list = predict_list + output.tolist()
                label_list   = label_list   + labels.tolist()

            # ここはtorch tensor →　python list　→ ndarrayに変換してます(はじめからnumpy配列で扱いましょう笑)
            predict_list = np.array(predict_list)
            label_list   = np.array(label_list)  
            calculate_score = Evaluation_icbhi(predict_list, label_list, self.save_folder, self.model_name)
            calculate_score.calculate_evaluation()
            calculate_score.auc_two_class(plot = 'True')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:1', help='set gpu device')
    parser.add_argument('--batch_size', type=int, default=256, help='set batch size')
    parser.add_argument('--feats_data_path', type=str, default='./data/feats', help='set save feats data path')
    parser.add_argument('--eval_dataset_id', type=int, help='set eval dataset_id')
    parser.add_argument('--model_name', type=str, default='model', help='set model name')
    args = parser.parse_args()

    device          = torch.device(args.device)
    batch_size      = args.batch_size
    data_root       = args.feats_data_path
    eval_dataset_id = args.eval_dataset_id
    model_name      = args.model_name

    num_classes = 4
    save_folder = f'./assets/dataset_{eval_dataset_id}/model'
    os.makedirs(save_folder, exist_ok = True)

    model     = Model().to(device)

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

    eval = EvaluateDNN(device     = device,
                       dataloader = dataloader,
                       model      = model,
                       save_folder= save_folder,
                       model_name = model_name)
    eval.evalate()


