export  CUDA_VISIBLE_DEVICES=2
mkdir -p model
mkdir -p log
nohup python cifar10_main.py --data_dir=data --model_dir=model > log/train.log 2>&1 &
