import os

GPU = 3
dataset = 'cifar100'
DROP = ['0.5', '0']
SEED = [1, 2, 3, 4, 5]

for seed in SEED:
    for dp in DROP:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session9.py --seed %d --dataset %s --drop_p %s' % (GPU, seed, dataset, dp)
        os.system(cmd)