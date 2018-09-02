import os

GPU = 0
dataset = 'cifar100'
SEED = [1, 2, 3]
std = 0
d = '0.02'
for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session16.py --seed %d --dataset %s --distill %s --rand_std %d' % (GPU, seed, dataset, d, std)
    os.system(cmd)