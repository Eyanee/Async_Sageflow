#!/bin/bash
cd ../Sageflow_code

# For mnist and fmnist, epoch 150
# For cifar10, epoch: 1200

python Sageflow_async.py --epoch 30 --update_rule Sageflow --poison_methods ourpoisonMethod --lrdecay 2000 --data_poison False --inverse_poison False --new_poison False  --model_poison True --dataset fmnist --frac 0.2 --attack_ratio 0.4 --gpu_number 1 --iid 1 --model_poison_scale 0.1 --eth 1 --delta 0.5  --lam 0.5 --seed 2021 --staleness 6 --scale_weight 40

