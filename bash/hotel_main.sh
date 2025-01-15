#!/bin/bash

# Running ESTAG experiments for subset hotel
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 1e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 2e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 3e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 4e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 5e-4 --epochs 60
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset hotel --lr 1e-3 --epochs 60
