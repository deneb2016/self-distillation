import os

GPU = 3
dataset = 'cifar10'
seed = 2
drop_p = '0.5'

cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session11.py --seed %d --dataset %s --drop_p %s --no_aug' % (GPU, seed, dataset, drop_p)
os.system(cmd)