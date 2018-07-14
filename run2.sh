CUDA_VISIBLE_DEVICES=1 python train_eval.py --stoch_depth 0.5 --s 1001 --seed 1 --dataset cifar100 --save_dir ../repo/distill/cifar100/resnet34/run1/
CUDA_VISIBLE_DEVICES=1 python train_eval.py --stoch_depth 0.5 --s 1001 --seed 1 --dataset cifar10 --save_dir ../repo/distill/cifar10/resnet34/run1/

