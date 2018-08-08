import os

GPU = 0
dataset = 'cifar10'
SEED = [1, 2, 3]
t = '1'
D = ['0.1', '0.02', '0.004']

for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session3.py --seed %d --dataset %s --temp %s --distill %s' % (GPU, seed, dataset, t, d)
        os.system(cmd)