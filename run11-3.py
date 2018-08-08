import os

GPU = 3
dataset = 'cifar100'
WD = ['0.0001', '0.00001']
SEED = [1, 2, 3, 4, 5]

for seed in SEED:
    for wd in WD:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session8.py --seed %d --dataset %s --wd %s' % (GPU, seed, dataset, wd)
        os.system(cmd)