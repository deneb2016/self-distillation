import os

GPU = 1
dataset = 'cifar10'
SEED = [2, 3, 4, 5]
for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session4.py --seed %d --dataset %s --drop_p 0 --feat_dim 4096 --wd 0.00001' % (GPU, seed, dataset)
    os.system(cmd)