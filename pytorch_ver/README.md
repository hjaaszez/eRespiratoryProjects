# Respiratory sound classification experiment by Pytorch

## Getting Started

### 1. Create a Docker Container

**1. Install Docker**
こちらの記事を参考にdockerのインストールをしてください。
https://qiita.com/tatsuya11bbs/items/3af03e704812b6c89965

**2. DockerfileからDocker Imageのビルド**
```
docker image build -t cls-experiment/pytorch .
```
注意: Dockerfileが存在するディレクトリにて上記コマンドを入力してください。

**3. Docker Containerの起動**
```
cd ../../
docker run -it -d --gpus '"device=0"' --name cls_env_torch -v ${PWD}/cls-experiment:/home/cls-experiment cls-experiment/pytorch:latest bash
```
各オプションは調べましょう。
- --gpus : gpuを指定。　device　id は nvidia-smi等で確認しましょう。
- -v     : ローカル環境の ```cls-experiment```をコンテナ上の```/home/cls-experiment```と共有。

**4. Docker Containerにattach**
```
docker attach cls_env_torch
```

detachする際は```cntl + P + Q```です。```cntl + D```するとコンテナが止まります。 


### 2. Install the libraries
コンテナにattach後
```
cd /home/cls-experiment/pytorch_ver
pip install -r requirements.txt
```


### 3. Run script
```
bash run.sh
```
