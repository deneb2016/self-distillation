import os

GPU = 3
dataset = 'cifar100'
SEED = [4, 5]

D = ['0.1', '0.02', '0.004']
t = '7'
wd = '0.0001'
for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session13.py --seed %d --dataset %s --wd %s --temp %s --distill %s' % (GPU, seed, dataset, wd, t, d)
        os.system(cmd)