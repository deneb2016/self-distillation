import os

GPU = 1
dataset = 'cifar100'
SEED = [1, 2, 3, 4, 5]

for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session14.py --seed %d --dataset %s' % (GPU, seed, dataset)
    os.system(cmd)