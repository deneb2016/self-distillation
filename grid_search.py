import os


for model_name in os.listdir(args.dir):
    if model_name.endswith('.pth'):
        os.system('python eval.py --model_path %s' % os.path.join(args.dir, model_name))

