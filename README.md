# Classification of respiratory sound
This repository manages respiratory sound classification experiments.

## Paper
This code is an implementation of the following paper.
```
N. Asatani et al., "Classification of Respiratory Sounds Using Improved Convolutional Recurrent Neural Network, Computers & Electrical Engineering, Vol. 94, pp.1-10, 2021. DOI=https://www.sciencedirect.com/science/article/pii/S0045790621003372
```

# Usage
Clone the repository with the following command

```
git clone https://github.com/kamilab-respiratory-sounds/cls-experiment.git
```

# Prepare Dataset
Copy the dataset with the following command
```
bash set_dataset.sh
```

# Select a Deep Learning Framework
If you use pytorch, move the folder to ```pytorch_ver```.  

If you use tensorflow, move the folder to ```tensorflow_ver```.

# File Structure
```
.
├── ICBHI_all # (Place the dataset here.)
├── pytorch_ver
│   ├── README.md
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run.sh
│   └── script
│       ├── 0_data_prep.py
│       ├── 1_feats_extract.py
│       ├── 2_train.py
│       ├── 3_test.py
│       ├── dataloader.py
│       └── model
│           └── CRNN.py
├── README.md
└── tensorflow_ver
    ├── README.md
    ├── Dockerfile
    ├── requirements.txt
    ├── run.sh
    └── script
        ├── 0_data_prep.py
        ├── 1_feats_extract.py
        ├── 2_train.py
        ├── 3_test.py
        ├── dataloader.py
        └── model
            └── CRNN.py
```
