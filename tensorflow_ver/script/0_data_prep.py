'''
音声データをn分割する。(実験では5分割を採用)
'''
import os
import glob
import shutil
import sys
import random
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

class MakeDataset():
    def __init__(self, split_num, data_dir, save_data_path, label_list):
        self.split_num      = split_num # 分割数
        self.data_dir       = data_dir  # ICBHIデータすべて(ICBHI_all)
        self.save_data_path = save_data_path #保存パス
        self.label_list     = label_list # 分類クラス
        self.sr             = 4000
        
        self.MakeDatasetDir() #保存用のディレクトリの作成

    def MakeDatasetDir(self):
        '''
        保存用ディレクトリの確保
        '''
        for dataset_id in range(self.split_num):
            save_dir = f'{self.save_data_path}/dataset_{dataset_id + 1}'
            for label in self.label_list:
                os.makedirs(f'{save_dir}/{label}',exist_ok=True)
    
    def SplitData(self):
        '''
        データの分割
        '''
        for label in self.label_list:
            data_paths = glob.glob(f'{self.data_dir}/{label}/*.wav') #各ラベルのすべてのwavパス格納
            random.shuffle(data_paths)
            divide_data_paths = [data_paths[i::self.split_num] for i in range(self.split_num)] #split_num数に分割            
            
            for i in range(self.split_num):
                for path in divide_data_paths[i]:
                    audio, sr = librosa.load(path, sr=self.sr)
                    self.SaveData(i, label, path, audio)

    def SaveData(self, dataset_id, label, path, audio):
        '''
        splitしたwavの保存
        '''
        save_path = f'{self.save_data_path}/dataset_{dataset_id + 1}/{label}/{os.path.basename(path)}'
        sf.write(save_path, audio, self.sr)
        print(save_path)        
        

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--split_num', type=int, default=5, help='set split number')
    parser.add_argument('--ICBHI_path', type=str, default='../ICBHI_all', help='set ICBHI dataset path')
    parser.add_argument('--save_data_path',type=str, default='./data/wav', help='set save data path')
    args = parser.parse_args()

    label_list   = ['normal', 'crackle', 'wheeze', 'crackle_wheeze'] 
    
    make_dataset = MakeDataset(split_num=args.split_num,
                               data_dir=args.ICBHI_path,
                               save_data_path=args.save_data_path,
                               label_list=label_list)
    make_dataset.SplitData()

