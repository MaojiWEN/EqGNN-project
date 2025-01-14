#!/bin/bash

# Running ESTAG experiments for subset zara1
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 1e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 2e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 3e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 4e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 5e-4 --epochs 80
CUDA_VISIBLE_DEVICES=0 python main_eth_ESTAG.py --subset zara1 --lr 1e-3 --epochs 80