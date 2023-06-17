import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
import numpy as np
from PIL import Image
import os, glob
import nibabel as nib
import cv2
import shutil
import random


from google.colab import drive
drive.mount('driveAndreu')


TFG = path = '/content/driveAndreu/MyDrive/TFG/'
eval_path = os.path.join(path, 'evaluation')            #Folder which contains 30 TC images for evaluation.
train_path = os.path.join(path, 'data_2')               # Folder which contains both, data and label of the training images.
imagePath = os.path.join(train_path, 'data')            # This folder contains (100 TC images) non completly preprocessed of human brains.
maskPath = os.path.join(train_path, 'label')            # This folder contains (100 masks of the previous images, also called groundtruth).

#All the files that I'll generate from the previous folder will be stored in Results
results = os.path.join(TFG, 'Results')

#Preprocessed nii images
preprocessed_img = os.path.join(results, 'Prerpocessed Images')


#Here I'll save the preprocessed images as slices
img_slices = os.path.join(results, 'Image Slices')
mask_slices = os.path.join(results, 'Mask Slices')

#Since there are lots of images without clear hemorrages or really small ones, here I'll store the images which have clear hemorrages
slices_with_hemorrhages = os.path.join(results, 'Slices with visible hemorrhages') #In this folder I will store only images which contain hemorrhages.
img_slices_hemo = os.path.join(slices_with_hemorrhages, 'data/data')
mask_slices_hemo = os.path.join(slices_with_hemorrhages, 'label/label')

#If I want to test the algorithm with all the images I'll need to keep the organiased for that
all_data_preprocessed = os.path.join(results, 'All Slices Preprocessed')
all_img_prep = os.path.join(all_data_preprocessed, 'data/data')
all_mask_org = os.path.join(all_data_preprocessed, 'label/label')

#Slices split according to patient
patient_img = os.path.join(results, 'Patient Slices/general/data/data')
patient_mask = os.path.join(results, 'Patient Slices/general/label/label')
patient_img_train = os.path.join(results, 'Patient Slices/train/data/data')
patient_mask_train = os.path.join(results, 'Patient Slices/train/label/label')
patient_img_test = os.path.join(results, 'Patient Slices/test/data/data')
patient_mask_test = os.path.join(results, 'Patient Slices/test/label/label')


#train
train = os.path.join(results, 'train')
train_img = os.path.join(train, 'data/data')
train_mask = os.path.join(train, 'label/label')
#test
test = os.path.join(results, 'test')
test_img = os.path.join(test, 'data/data')
test_mask = os.path.join(test, 'label/label')

#Independent test
independent_test = os.path.join(results, 'Patient Slices/Independent test')









# Find all .nii images in the imagesPath folder
image_files = [f for f in os.listdir(imagePath) if f.endswith('.nii')]

    # Threshold each image and save the results
for image_file in image_files:
  # Load the image
    img = nib.load(os.path.join(imagePath, image_file)).get_fdata()

# Apply the threshold to remove the skull
    img[img < -10] = -2000 #removing parts of the image out of the brain
    img[img > 100] = -2000 #I try to remove every bit of skull and noise that I won't focus on
    img[(img > 50) & (img < 100)] *= 2.5 #since the hemorrhages are typically between this values I try to enchance this part of the image
    img[img <50] *= 0.2 #Trying to lower all sections under 50 in order to be have a better contrast


# Save the processed image in the no_skull directory
    processed_image = nib.Nifti1Image(img, np.eye(4))
    nib.save(processed_image, os.path.join(preprocessed_img, image_file))
        
    print(f'{image_file} processed and saved in {preprocessed_img}')
