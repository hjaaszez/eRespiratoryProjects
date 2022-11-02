# Respiratory sound classification experiment by TensorFlow(1.13)

基本はtorchで開発することをお勧めします。
先輩の化石コードを動かす場合、こちらを参考に実行してみてください。

## Getting Started

### 1. Create a Docker Container

**1. Install Docker**
こちらの記事を参考にdockerのインストールをしてください。
https://qiita.com/tatsuya11bbs/items/3af03e704812b6c89965

**2. Docker HubからTensorFlow-GPU 1.13のdocker imageをpull**
```
docker pull tensorflow/tensorflow:1.13.2-gpu-py3
```

**3. Docker Containerの起動**
```
cd ../../
docker run -it -d --gpus '"device=0"' --name cls_env_tf -v ${PWD}/cls-experiment:/home/cls-experiment tensorflow/tensorflow:1.13.2-gpu-py3 bash
```
各オプションは調べましょう。
- --gpus : gpuを指定。　device　id は nvidia-smi等で確認しましょう。
- -v     : ローカル環境の ```cls-experiment```をコンテナ上の```/home/cls-experiment```と共有。

**4. Docker Containerにattach**
```
docker attach cls_env_tf
```

detachする際は```cntl + P + Q```です。```cntl + D```するとコンテナが止まります。 


### 2. Install the libraries
コンテナにattach後
```
cd /home/cls-experiment/tensorflow_ver
pip install -r requirements.txt
```


### 3. Run script
```
bash run.sh
```
