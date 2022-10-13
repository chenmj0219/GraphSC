#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python pretrain_mgsc.py --lr 0.001 --alpha 0.01 --beta 2.0 --aug random1 --rate1 0.25 --rate2 0.3