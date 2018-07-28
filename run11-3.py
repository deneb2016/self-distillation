import os

GPU = 3
dataset = 'cifar10'
SEED = [1, 2, 3, 4, 5]
save_dir = '../repo/distill/%s/vgg16do/session4/' % dataset
for seed in SEED:
    cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session4.py --seed %d --dataset %s --save_dir %s --drop_p 0.5 --feat_dim 4096 --wd 0' % (GPU, seed, dataset, save_dir)
    os.system(cmd)