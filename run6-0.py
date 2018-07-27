# import os
#
# GPU = 0
# dataset = 'cifar10'
# T = ['1', '9']
# D = ['1', '0.2', '0.04', '0.008']
# seed = 5
# save_dir = '../repo/distill/%s/resnet34sd/session1/' % dataset
#
# for t in T:
#     for d in D:
#         cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
#         os.system(cmd)

import os

GPU = 0
dataset = 'cifar100'
T = ['1']
D = ['0.1', '0.02', '0.004', '0.5']
seed = 3
save_dir = '../repo/distill/%s/resnet34sd/session3/' % dataset

for t in T:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session3.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
        os.system(cmd)