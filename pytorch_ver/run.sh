# 実行スクリプト
# stage0: データ分割 ICBHI_allから交差検証用にデータ分割
# stage1 : 特徴抽出
# stage2 : DNN 学習
# stage3 : モデルの検証

# 止めたいところで適宜exitしましょう

echo 'start process'

split_num=5
ICBHI_data=../ICBHI_all
raw_data_path=./data/wav
feats_data_path=./data/feats

stage=0

if [ $stage = 0 ]; then 
    echo 'stage 0'
    if [ -d ${raw_data_path} ]; then
        echo 'Audio file has been splited, Skip this process.'
    else
        echo 'stage 0: split audio file'
        python3 script/0_data_prep.py --split_num ${split_num} --ICBHI_path ${ICBHI_data} --save_data_path ${raw_data_path}
        echo 'Finished split audio file'
    fi
    stage=1
fi

if [ $stage = 1 ]; then
    echo 'stage 1'
    if [ -d ${feats_data_path} ]; then
        echo 'Feature Extraction has been finished, Skip this process'
    else
        echo 'stage1: feature extraction'
        python3 script/1_feats_extract.py --split_num ${split_num} --raw_data_path ${raw_data_path} --feats_data_path ${feats_data_path}
        echo 'Finished feature extraction'
    fi
    stage=2
fi

device='cuda:1'
batch_size=256
num_epochs=5
model_name='CRNN'

if [ $stage = 2 ]; then
    echo 'stage 2'
    for dataset_id in 1 2 3 4 5; do

        assets_path=./assets/dataset_${dataset_id}
        echo 'stage2: Start training a deep learning model dataset_'${dataset_id}
        python3 script/2_train.py --device ${device} --batch_size ${batch_size} --num_epochs ${num_epochs} --feats_data_path ${feats_data_path} --eval_dataset_id ${dataset_id} --model_name ${model_name}
        echo 'Finished model training dataset_'${dataset_id}
    
    done
    stage=3
fi

if [ $stage = 3 ]; then
    echo 'stage 3'
    for dataset_id in 1 2 3 4 5; do

        assets_path=./assets/dataset_${dataset_id}        
        echo 'stage3: Evaluate a deep learning model dataset_'${dataset_id}
        python3 script/3_test.py --device ${device} --batch_size ${batch_size} --feats_data_path ${feats_data_path} --eval_dataset_id ${dataset_id} --model_name ${model_name}
        echo 'Finished experiment! dataset_'${dataset_id}
        
    done
fi

echo 'Finished all process'