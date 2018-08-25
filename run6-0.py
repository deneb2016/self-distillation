import os

GPU = 0
dataset = 'cifar10'
SEED = [1, 2, 3, 4, 5]

D = ['0.1', '0.02', '0.004']
t = '1'
wd = '0.0001'
for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session13.py --seed %d --dataset %s --wd %s --temp %s --distill %s' % (GPU, seed, dataset, wd, t, d)
        os.system(cmd)