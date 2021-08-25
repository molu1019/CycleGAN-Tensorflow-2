## Code obtained from https://github.com/italojs/resize_dataset_pascalvoc

import os
import argparse
from resize_images_pascalvoc.utils import create_path, add_end_slash
from resize_images_pascalvoc.image import process_image
from optparse import OptionParser

## no need for argpaser due to internal function call
'''parser = argparse.ArgumentParser()

parser.add_argument(
    '-p',
    '--path',
    dest='dataset_path',
    help='Path to dataset data ?(image and annotations).',
    required=True
)
parser.add_argument(
    '-o',
    '--output',
    dest='output_path',
    help='Path that will be saved the resized dataset',
    default='./',
    required=True
)
parser.add_argument(
    '-x',
    '--new_x',
    dest='x',
    default=256,
    help='The new x images size',
    required=True
)
parser.add_argument(
    '-y',
    '--new_y',
    default=256,
    dest='y',
    help='The new y images size',
    required=True
)
parser.add_argument(
    '-s',
    '--save_box_images',
    dest='save_box_images',
    help='If True, it will save the resized image and a drawed image with the boxes in the images',
    default=0
)
'''

IMAGE_FORMATS = ('.jpeg', '.JPEG', '.png', '.PNG', '.jpg', '.JPG')

#args = parser.parse_args()

def resize_label(save_box_images, dataset_path, output_path1, new_x, new_y):
    create_path(output_path1)
    if int(save_box_images):
        create_path(''.join([output_path1, '/boxes_images']))
    
    dataset_path = add_end_slash(dataset_path)
    output_path = add_end_slash(output_path1)
    
    for root, _, files in os.walk(dataset_path):
        output_path = os.path.join(output_path, root[len(dataset_path):])
        create_path(output_path)

        for file in files:
            if file.endswith(IMAGE_FORMATS):
                file_path = os.path.join(root, file)
                process_image(file_path, output_path, int(new_x),int(new_y), save_box_images)
