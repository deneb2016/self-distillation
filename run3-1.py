import os

GPU = 1
dataset = 'cifar100'
SEED = [4, 5]
t = '5'
D = ['0.1', '0.02', '0.004']

for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session3.py --seed %d --dataset %s --temp %s --distill %s' % (GPU, seed, dataset, t, d)
        os.system(cmd)