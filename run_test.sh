#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python3 test.py \
	--experiment_dir ./output/powertrain_all \
	--batch_size 4
