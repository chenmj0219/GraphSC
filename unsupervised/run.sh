#!/bin/sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -q gpu2
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J graph



CUDA_VISIBLE_DEVICES=0 python GGG.py --DS REDDIT-BINARY --hidden-dim 32  \
--lr 0.01 --num-gc-layers 3 --lambda1 1  --lambda2 1 --lambda3 0.01 --epochs 20 \
--aug random2 --rate1 0.05 --rate2 0.2 --seeds 1
