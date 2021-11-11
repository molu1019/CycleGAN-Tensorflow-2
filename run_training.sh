#!/bin/bash
     
CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
	--dataset powertrain_all \
	--batch_size 2 \
	--epochs 50 \
	--gradient_penalty_mode wgan-gp \
	--gradient_penalty_weight 0 \
	--identity_loss_weight 5 \
	--adversarial_loss_mode lsgan \
	--load_size 420 \
	--crop_size 384
