This repository contains the PyTorch implementation for "VFLMG" as well as other baseline methods for MSN 2023 paper entitled "Adaptive and Efficient Participant Selection in Vertical Federated Learning". Our goal is to facilitate the reproduction of the results presented in the paper. Our implementations leverage the public recommendation library, mixed_KSG. More details about mixed_KSG can be found [here](https://github.com/wgao9/mixed_KSG).

## Setup
```
pip install -r requirements.txt
```

## Train
To generate the participants selected by VFLMG, run:
```
python Greedy.py
```
To obtain the final results, execute:
```
python MultipleClient.py
```