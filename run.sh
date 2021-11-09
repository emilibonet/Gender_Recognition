#!/bin/bash
srun --time 1:00:00 --gres=gpu:1,gpumem:20G python3 detect.py