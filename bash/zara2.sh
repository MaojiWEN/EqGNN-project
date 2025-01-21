#!/bin/bash

# Running ESTAG experiments for subset zara2
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset zara2 --lr 4e-4 --epochs 100
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset zara2 --lr 5e-4 --epochs 100
CUDA_VISIBLE_DEVICES=0 python main_eth_stft.py --subset zara2 --lr 1e-3 --epochs 100