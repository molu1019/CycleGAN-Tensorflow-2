"""Data Augmentaion for scaling n.i.O images from powertrain"""

import argparse
from operator import index
import os, cv2, random
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import glob
import PIL
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
def augmentation(dataset):
    """Augment all png or jpg images in a folder according to the args that have been passed by the terminal command

	Args:
		path
	
	Returns:
		Augmented files in the specified folder
	"""	 
    dataset_path = dataset
    save_directory = dataset

    # Keras Imagedatagenerator for automatic augmentation
    datagen = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, rescale=1./255,
            shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode='nearest')

    '''try:
        os.mkdir(augmented_images)
    except OSError:
        print ("Creation of the directory %s failed" % augmented_images)
    else:
        print("Successfully created the directory %s " % augmented_images)'''

    # define parameters
    #files = os.listdir(dataset_path)
    pngs = glob.glob(os.path.join(dataset_path, '*.jpg')) + glob.glob(os.path.join(dataset_path, '*.png'))
    # define number of loops
    i=1
    aug_images = 1
    numb_images = len(pngs)*aug_images
    for index, j in enumerate(pngs):
        img = tf.keras.preprocessing.image.load_img(j)
        x = tf.keras.preprocessing.image.img_to_array(img)
        # x.shape returns[number of images, height, width, rgb channels] = [1, 1080, 1920, 3] 
        x = x.reshape((1,) + x.shape)
        i = 1
        for batch in datagen.flow(x, batch_size = 1,
                                         save_to_dir = save_directory, 
                                         save_prefix = "aug_%s" %(index),
                                         save_format = 'png'):
            
            i = i+1
            if i > aug_images:
                break
    print("Successfully created %s augmented Images" % numb_images)    
    

def data_augmentation(dataset, factor):
    parser = argparse.ArgumentParser(description= 'Data augmentation for training with images')
    parser.add_argument('--dataset', type=str, help='Directory where images will be augmented')
    args = parser.parse_args()

    try:
        augmentation(args.dataset)
    except FileNotFoundError:
	    print("Wrong file/folder path")
    else:
	    print("augmentation.py was successful")