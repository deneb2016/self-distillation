import os

GPU = 3
dataset = 'cifar100'
SEED = [1, 2, 3]

D = ['0.1', '0.02', '0.004']
t = '7'
for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session14.py --seed %d --dataset %s --temp %s --distill %s  --drop_last_only' % (GPU, seed, dataset, t, d)
        os.system(cmd)