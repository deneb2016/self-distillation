import os

GPU = 0
dataset = 'cifar10'
FEAT_DIM = ['4096']
DROP_P = ['0.3']

seed = 1
save_dir = '../repo/distill/%s/resnet34sd/session4/' % dataset

for fd in FEAT_DIM:
    for dp in DROP_P:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session4.py --seed %d --dataset %s --save_dir %s --feat_dim %s --drop_p %s' % (GPU, seed, dataset, save_dir, fd, dp)
        os.system(cmd)
