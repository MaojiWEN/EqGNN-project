#!/bin/bash

# Running ESTAG experiments for subset eth
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 1e-5 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 2e-5 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 3e-5 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 4e-5 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 5e-5 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 1e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 2e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 3e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 4e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset eth --lr 5e-4 --epochs 80
