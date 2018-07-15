import os

GPU = 1
dataset = 'cifar10'
T = ['9', '11', '13', '15']
D = ['1', '0.2', '0.04', '0.008']
seed = 1
save_dir = '../repo/distill/%s/resnet34sd/session1/' % dataset

cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp 1 --distill 0' % (GPU, seed, dataset, save_dir)
os.system(cmd)

for t in T:
    for d in D:
        cmd = 'CUDA_VISIBLE_DEVICES=%d python train_eval_session1.py --seed %d --dataset %s --save_dir %s --stoch_depth 0.5 --temp %s --distill %s' % (GPU, seed, dataset, save_dir, t, d)
        os.system(cmd)

