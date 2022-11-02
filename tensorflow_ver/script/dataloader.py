import glob
import os
import numpy as np
from PIL import Image
import librosa
import random
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler

def set_dir(id, dataset_dir, train=True):
    dir_paths = []        
    split_num = len(os.listdir(dataset_dir))
    
    if train == True:
        #Trainデータのパスをdir_pathsに格納
        for i in range(split_num):
            if i  == id - 1:
                pass
            else:
                dir_paths.append(f'{dataset_dir}/dataset_{i + 1}/')
    else:
        #Testデータのパスをdir_pathsに格納
        dir_paths.append(f'{dataset_dir}/dataset_{id}/')
    
    return dir_paths

def dataloader(dir_paths, classes = ["normal", "crackle", "wheeze", "crackle_wheeze"], img_height=40, img_width=40):
    spectrogram = [] # データ格納
    label       = [] # ラベル格納
    data_name   = [] # データ名格納
    
    for data_dir in dir_paths:
        for index, name in enumerate(classes):
            dir = data_dir + name
            files = glob.glob(f"{dir}/*.png")
            for i, file in enumerate(files):
                image = Image.open(file)
                # RGBA(4ch)からRGB(3ch)への変換.
                image = image.convert("RGB")
                image = image.resize((img_width, img_height))
                data = np.asarray(image)
                
                spectrogram.append(data)
                label.append(index)
                data_name.append(os.path.basename(file).rstrip(".png"))

    spectrogram = np.array(spectrogram)
    label       = np.array(label)
    data_name   = np.array(data_name)

    spectrogram = spectrogram.astype("float32")
    spectrogram = spectrogram / 255.0

    # one hot encoding
    label = np_utils.to_categorical(label, 4)
 
    return spectrogram, label, data_name