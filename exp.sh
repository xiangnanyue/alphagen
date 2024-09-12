#!/bin/bash

python train_maskable_ppo.py --seed=123 --pool=10 --code=all --steps=250000 --freq=60min --pred_len=8
