from keras.models import model_from_json, Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import os
import glob
import pandas
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from dataloader import dataloader, set_dir
from evaluation import Evaluation_icbhi

def eval(data_root, eval_dataset_id, batch_size, model, save_folder, model_name):  

    '''
    data load
    '''
    x_test_1, y_test_1, z  = dataloader(set_dir(eval_dataset_id, dataset_dir=data_root, train=False))
    
    '''
    prediction
    '''
    y_preds = model.predict(x_test_1, verbose = 1, batch_size = batch_size)

    evaluation = Evaluation_icbhi(predict_label=y_preds, true_label=y_test_1, result_dir = save_folder, result_name = model_name)
    evaluation.calculate_evaluation()
    evaluation.auc_four_class()
    evaluation.auc_two_class(plot=True)


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256, help='set batch size')
    parser.add_argument('--feats_data_path', type=str, default='./data/feats', help='set save feats data path')
    parser.add_argument('--eval_dataset_id', type=int, help='set eval dataset_id')
    parser.add_argument('--model_name', type=str, default='model', help='set model name')
    args = parser.parse_args()

    batch_size      = args.batch_size
    data_root       = args.feats_data_path
    eval_dataset_id = args.eval_dataset_id
    model_name      = args.model_name

    save_folder = f'./assets/dataset_{eval_dataset_id}/model'
    os.makedirs(save_folder, exist_ok = True)

    json_string = open(f'{save_folder}/{model_name}.json').read()
    model = model_from_json(json_string)
    model.load_weights(f'{save_folder}/{model_name}.hdf5')

    eval(data_root       = data_root,
         eval_dataset_id = eval_dataset_id,
         batch_size      = batch_size,
         model           = model,
         save_folder     = save_folder,
         model_name      = model_name)



