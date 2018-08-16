import os

GPU = 0
dataset = 'cifar100'
seed = 1
drop_p = '0.5'

cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session10.py --seed %d --dataset %s --drop_p %s --no_aug' % (GPU, seed, dataset, drop_p)
os.system(cmd)