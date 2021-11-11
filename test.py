from posixpath import join
from sys import path
import cv2
from skimage.io.collection import ImageCollection
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import glob
import data
import module
#import sys
#sys.path.append('..')
import resize_images_pascalvoc
from resize_images_pascalvoc.resize_main import resize_label


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir')
py.arg('--batch_size', type=int, default=32)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                   label resize                             =
# ==============================================================================

dataset_path = py.join(args.datasets_dir, args.dataset, 'testA')
output_path = py.join(args.datasets_dir, args.dataset, 'testA')
#py.mkdir(output_path)
# uncomment to resize and copy xml labels
#save_box_images = True
#resize_label(save_box_images, dataset_path, output_path, new_x=256, new_y=256)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = (py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.png')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.PNG')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.JPG'))
B_img_paths_test = (py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.png')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.JPG'))

#
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0

for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1

import imageio
import os
import shutil

save_FID = py.join(args.experiment_dir, 'samples_testing', 'A2B_FID')
py.mkdir(save_FID)
i = 0
# Copy XML labels
#src = py.join(args.datasets_dir, args.dataset, 'testA')

#for jpgfile in glob.glob(os.path.join(src, "*.json")):
#    shutil.copy(jpgfile, save_FID)
#try:
#    shutil.copyfile(src, save_FID)
#    print("json labels copied successfully.")
# for general errors
#except:
#    print("Error occurred while copying file.")

# save Image as single image B for FID calculation
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        imgB = A2B_i.numpy()
        imageio.imwrite(os.path.join(save_FID ,(py.name_ext(A_img_paths_test[i]))), imgB)
        i += 1


save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A')
py.mkdir(save_dir)
i = 0
for B in B_dataset_test:
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i in zip(B, B2A, B2A2B):
        img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
        im.imwrite(img, py.join(save_dir, py.name_ext(B_img_paths_test[i])))
        i += 1
