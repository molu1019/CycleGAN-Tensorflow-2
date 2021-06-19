Data Augmentation

import os, glob, cv2
import keras_preprocessing


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# origin path
images_path = glob('/home/molu1019/workspace/CycleGAN-Tensorflow-2/datasets/powertrain/trainA/*.png')
# define the name of the directory to be created
augmented_path = glob('/home/molu1019/workspace/CycleGAN-Tensorflow-2/datasets/powertrain/aug_trainA')
# list to store paths of images
images = []  

try:
    os.mkdir(augmented_path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


    
for j in images_path:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)