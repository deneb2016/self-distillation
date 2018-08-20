import os

GPU = 2
dataset = 'cifar100'
SEED = [1, 2, 3]

D = ['0.1', '0.02', '0.004']
t = '13'
wd = '0.0001'
for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session12.py --seed %d --dataset %s --drop_last_only --wd %s --temp %s --distill %s' % (GPU, seed, dataset, wd, t, d)
        os.system(cmd)