import os

GPU = 0
dataset = 'cifar100'
SEED = [1, 2, 3, 4, 5]
for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session4.py --seed %d --dataset %s --drop_p 0.5 --feat_dim 2048 --wd 0.0001' % (GPU, seed, dataset)
    os.system(cmd)