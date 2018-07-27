# import os
#
# GPU = 3
# dataset = 'cifar100'
# T = ['9', '11', '13', '15']
# D = ['1', '0.2', '0.04', '0.008']
# seed = 1
# save_dir = '../repo/distill/%s/resnet34sd/session1/' % dataset
#
# cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp 1 --distill 0' % (GPU, seed, dataset, save_dir)
# os.system(cmd)
#
# for t in T:
#     for d in D:
#         cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
#         os.system(cmd)

# import os
#
# GPU = 3
# dataset = 'cifar100'
# FROM = ['31', '101']
# D = ['1', '0.2', '0.04']
# seed = 2
# save_dir = '../repo/distill/%s/resnet34sd/session2/' % dataset
#
# for df in FROM:
#     for d in D:
#         cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session2.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp 9 --distill_from %s --distill %s' % (GPU, seed, dataset, save_dir, df, d)
#         os.system(cmd)
#

import os

GPU = 3
dataset = 'cifar10'
T = ['1', '3', '5', '7']
D = ['1', '0.2', '0.04', '0.008']
seed = 4
save_dir = '../repo/distill/%s/resnet34sd/session1/' % dataset

cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 1 --temp 0 --distill 0' % (GPU, seed, dataset, save_dir)
os.system(cmd)

for t in T:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
        os.system(cmd)


