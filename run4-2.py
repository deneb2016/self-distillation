import os

GPU = 2
dataset = 'cifar10'
SEED = [1, 2, 3]
dp = '0.0'
wd = '0.0001'

for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session10.py --seed %d --dataset %s --drop_p %s --wd %s' % (GPU, seed, dataset, dp, wd)
    os.system(cmd)