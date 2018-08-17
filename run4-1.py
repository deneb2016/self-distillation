import os

GPU = 1
dataset = 'cifar10'
SEED = [1, 2, 3, 4, 5]
drop_p = '0.6'

for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session4.py --seed %d --dataset %s --drop_p %s' % (GPU, seed, dataset, drop_p)
    os.system(cmd)