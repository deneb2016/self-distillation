import os

GPU = 0
dataset = 'cifar100'
T = ['15']
D = ['0.1', '0.02', '0.004', '0.5']
seed = 3
save_dir = '../repo/distill/%s/resnet34sd/session3/' % dataset

for t in T:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session3.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
        os.system(cmd)