# Presto & WorldCereal

## How to run
Install dependencies:
```bash
pip install -r requirements.txt
```
Follow instructions in `data/README.md` to get data. 

Run MAE pretraining from random init:
```bash
python train.py
```
Run MAE pretraining starting from pretrained (on Presto data) params:
```bash
python train.py --warm_start
```
To skip training and only run evaluation, use `--n_epochs 0` and `--warm_start`:
```bash
python train.py --n_epochs 0 --warm_start
```