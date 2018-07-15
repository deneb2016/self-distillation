import os

GPU = 0
session = 10001
dataset = 'cifar10'
T = ['1', '3', '5', '7', '9', '11', '13', '15']
D = ['1', '0.2', '0.04', '0.008']
seed = 1

for t in T:
    for d in D:
        save_dir = '../repo/distill/%s/resnet34/seed%d/' % (dataset, seed)
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval.py --s %d --seed %d --dataset %s --save_dir %s' % (GPU, session, seed, dataset, save_dir)
        os.system(cmd)
