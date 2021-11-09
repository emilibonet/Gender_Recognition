#!bin/bash
srun --time 0:30:00 --gres=gpu:1,gpumem:5G python3 detect.py