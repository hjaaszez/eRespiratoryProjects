'''
音声データから特徴量(Spectrogram)を抽出→画像(png)として抽出

一般的には画像変換は行わず、numpy配列等で保存することが多いが、画像として扱うことでwavごとの音声長さの違いを対処した．(基本はゼロ埋め)
また、dataloaderで音声特徴抽出をオンザフライで行う方が一般的だが、今回は一度特徴量ファイルを保存することにする．
'''

import os
import glob
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

class FeatureExtract():
    '''
    Feature ExtractにてSTFT抽出
    '''

    def __init__(self, split_num, raw_data_path, feats_data_path, label_list):
        self.split_num       = split_num
        self.raw_data_path   = raw_data_path #./data/wav
        self.feats_data_path = feats_data_path # ./data/feats
        self.label_list      = label_list
        
        '''
        STFT用のパラメータ
        '''
        self.sr         = 4000
        self.n_fft      = 0.025 # window size
        self.hop_length = 0.010 # overlab size

        self.MakeDatasetDir()

    def MakeDatasetDir(self):
        '''
        保存用ディレクトリの確保
        '''
        for dataset_id in range(self.split_num):
            save_dir = f'{self.feats_data_path}/dataset_{dataset_id + 1}'
            for label in self.label_list:
                os.makedirs(f'{save_dir}/{label}',exist_ok=True)

    # Feature Extraction
    def CalculateSTFT(self, audio):
        return librosa.stft(audio, n_fft=librosa.time_to_samples(self.n_fft, self.sr),hop_length=librosa.time_to_samples(self.hop_length, self.sr))
    
    def SaveSTFT(self, dataset_id, label, wav_path):
        save_path = f'{self.feats_data_path}/dataset_{dataset_id + 1}/{label}/{os.path.splitext(os.path.basename(wav_path))[0]}.png'
        
        '''
        音声のロードとSTFTの算出
        '''
        audio, sr = librosa.load(wav_path, sr=self.sr)
        stft = self.CalculateSTFT(audio)
        
        fig = plt.figure(figsize=(224, 224),dpi=1)
        fig.add_axes([0, 0, 1, 1])  # [x0, y0, width(比率), height(比率)]
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max), sr=self.sr, x_axis='time', y_axis='linear',cmap='magma')  # スペクトログラムを表示
        plt.ylim(0,2000)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        print(save_path)
    
    def MakeFeatsDataset(self):
        for dataset_id in range(self.split_num):
            for label in self.label_list:
                wav_paths = glob.glob(f'{self.raw_data_path}/dataset_{dataset_id + 1}/{label}/*.wav')
                for wav_path in wav_paths:
                    self.SaveSTFT(dataset_id, label, wav_path)

    
if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--split_num', type=int, default=5, help='set split number')
    parser.add_argument('--raw_data_path', type=str, default='./data/wav', help='set raw data path')
    parser.add_argument('--feats_data_path', type=str, default='./data/feats', help='set save feats data path')
    args = parser.parse_args()

    label_list   = ['normal', 'crackle', 'wheeze', 'crackle_wheeze'] 

    feats_extraction = FeatureExtract(split_num=args.split_num,
                                      raw_data_path=args.raw_data_path,
                                      feats_data_path=args.feats_data_path,
                                      label_list=label_list)

    feats_extraction.MakeFeatsDataset()