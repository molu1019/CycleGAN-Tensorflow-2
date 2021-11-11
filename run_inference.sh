#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3 inference.py \
	--experiment_dir /mnt/robolab/data/Bilddaten/GAN_train_data_sydavis-ai/Inference_Data/GAN_Inferences/pt19_1600syn_100re_50e \
	--batch_size 1
