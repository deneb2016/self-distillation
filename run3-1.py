import os

GPU = 1
dataset = 'cifar100'
SEED = [2, 4]
t = '3'
D = ['0.2', '0.04', '0.008']

for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session11.py --seed %d --dataset %s --temp %s --distill %s' % (GPU, seed, dataset, t, d)
        os.system(cmd)