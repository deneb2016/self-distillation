import os

GPU = 0
dataset = 'cifar10'
SEED = [1, 2, 3]
dp = '0'
wd = '0'

for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session10.py --seed %d --dataset %s --drop_p %s --wd %s --no_aug' % (GPU, seed, dataset, dp, wd)
    os.system(cmd)