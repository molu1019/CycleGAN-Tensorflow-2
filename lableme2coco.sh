#!/bin/bash
     
python3 labelme2coco.py \
	/mnt/robolab/data/Bilddaten/GAN_train_data_sydavis-ai/SydavisAI/eval \
	/mnt/robolab/data/Bilddaten/GAN_train_data_sydavis-ai/SydavisAI/eval_coco \
	--labels labels.txt
