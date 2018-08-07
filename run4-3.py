import os

GPU = 3
DATA = ['cifar100', 'cifar10']
feat_dim = 2048
SEED = [1, 2, 3, 4, 5]

for seed in SEED:
    for dataset in DATA:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session8.py --seed %d --dataset %s --feat_dim %d' % (GPU, seed, dataset, feat_dim)
        os.system(cmd)