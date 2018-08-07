import os

GPU = 2
dataset = 'cifar10'
SEED = [1, 2, 3, 4, 5]
t = 15
D = ['0.5', '0.1', '0.02', '0.004']

for seed in SEED:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session7.py --seed %d --dataset %s --temp %d --distill %s' % (GPU, seed, dataset, t, d)
        os.system(cmd)