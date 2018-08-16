import os

GPU = 2
dataset = 'cifar10'
seed = 1
drop_p = '0.3'

cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session11.py --seed %d --dataset %s --drop_p %s --no_aug' % (GPU, seed, dataset, drop_p)
os.system(cmd)