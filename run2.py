import os

GPU = 1
dataset = 'cifar10'
SEED = [1, 2, 3, 4, 5]
t = '1'
d = '0'
for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session11.py --seed %d --dataset %s --temp %s --distill %s' % (GPU, seed, dataset, t, d)
    os.system(cmd)