import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class RespireSoundDataset(Dataset):
    '''
    ICBHI Dataset スペクトログラムロード用のclass
    argument:
        data_root       : データセットのルートディレクトリ(```./data/feats)
        eval_dataset_id : 検証用データセット
            def SetDir() ->  訓練用データのパスを返す
        transform
        trainable       : Bool 訓練時はTrue, 検証時はFalse
        
    '''
    def __init__(self, data_root, eval_dataset_id, transform, trainable):
        self.data_root       = data_root
        self.eval_dataset_id = int(eval_dataset_id)
        self.transform       = transform
        self.trainable       = trainable
     
        self.dataset_dir = self.SetDir()
        self.label_list = {'normal':0, 'crackle':1, 'wheeze':2, 'crackle_wheeze':3}
        
    def SetDir(self):
        dir_paths = []        
        split_num = len(os.listdir(self.data_root))

        if self.trainable is True:
            #Trainデータのパスをdir_pathsに格納
            for i in range(split_num): 
                if i  == self.eval_dataset_id - 1:
                    # 検証用データセットはdir_pathsに格納しない
                    pass
                else:
                    for filepath in glob.glob(f'{self.data_root}/dataset_{i + 1}/*/*.png'):
                        dir_paths.append(filepath)

        else:
            #Testデータのパスをdir_pathsに格納
            dir_paths = glob.glob(f'{self.data_root}/dataset_{self.eval_dataset_id}/*/*.png')
        
        return dir_paths


    def __len__(self):
        return len(self.dataset_dir)

    def __getitem__(self, idx):
        path        = self.dataset_dir[idx]
        spectrogram = Image.open(path)
        spectrogram = spectrogram.convert('RGB')

        #spectrogram = spectrogram.permute(2,0,1)
        label_key   = path.split('/')[-2]
        label       = self.label_list[label_key]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

