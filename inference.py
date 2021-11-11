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
# =                                   inference                                =
# ==============================================================================

# data
A_img_paths_test = (py.glob(py.join(args.datasets_dir, args.dataset, 'inference'), '*.jpg')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'inference'), '*.png')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'inference'), '*.PNG')+
                    py.glob(py.join(args.datasets_dir, args.dataset, 'inference'), '*.JPG'))

#
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# restore
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    return A2B



# run
import imageio
import os
import shutil

save_inference = py.join(args.experiment_dir, 'Inference', str(args.adversarial_loss_mode)+"_Size"+str(args.crop_size)+"_Epo"+str(args.epochs))
py.mkdir(save_inference)
i = 0
dim = (1920, 1080)

# save Image as single image B for FID calculation
for A in A_dataset_test:
    A2B = sample_A2B(A)
    for A_i, A2B_i in zip(A, A2B):
        imgB = A2B_i.numpy()
        img_resized = cv2.resize(imgB, dim, interpolation = cv2.INTER_LANCZOS4)
        imageio.imwrite(os.path.join(save_inference ,(py.name_ext(A_img_paths_test[i]))), img_resized)
        i += 1

def resize_upsampling(old_folder, new_folder, size):
    dim = (1920, 1080)
    for image in os.listdir(old_folder):
        img = cv2.imread(os.path.join(old_folder, image))
        # INTER_CUBIC or INTER_LANCZOS4
        img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
        print('Shape: '+str(img.shape)+' is now resized to: '+str(img_resized.shape))
        cv2.imwrite(os.path.join(new_folder , image),img_resized)


