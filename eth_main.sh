#!/bin/bash

# Running ESTAG experiments for subset eth
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 1e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 2e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 3e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 4e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 5e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset eth --lr 1e-3 --epochs 60
