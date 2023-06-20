#First of all I import all the necessary libraries
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
import numpy as np
from PIL import Image
import glob
import nibabel as nib
import cv2
import shutil
import random

#Here I mount my drive
from google.colab import drive
drive.mount('driveAndreu')

#Now I create all the folders that I used during the proyect, u'll see that I declared folders that are not used in the code, but I used them manually in some cases.
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



#Now I begin with the thresholding

# Find all .nii images in the imagesPath folder
image_files = [f for f in os.listdir(imagePath) if f.endswith('.nii')]

# Threshold each image and save the results
for image_file in image_files:
  # Load the image
    img = nib.load(os.path.join(imagePath, image_file)).get_fdata()

# Apply the threshold to remove the skull
    img[img < -10] = -2000 #removing parts of the image out of the brain
    img[img > 100] = -2000 #I try to remove every bit of skull and noise that I won't focus on
    img[(img => 50) & (img <= 100)] *= 2.5 #since the hemorrhages are typically between these values I try to enhance this part of the image
    img[img <50] *= 0.2 #Trying to lower all sections under 50 in order to have a better contrast


# Save the processed image in the no_skull directory
    processed_image = nib.Nifti1Image(img, np.eye(4))
    nib.save(processed_image, os.path.join(preprocessed_img, image_file))
        
    print(f'{image_file} processed and saved in {preprocessed_img}')


# Find all .nii images in the images folder and show the 12th slice (I do this because the lesion usually has a part in this slice number on the patients)
imageFiles = os.listdir(preprocessed_img)
for imageFile in imageFiles:
    
    img = nib.load(os.path.join(preprocessed_img, imageFile)).get_fdata()
    imgSlice = img[:, :, 12]
    plt.imshow(imgSlice, cmap='gray')
    plt.title(os.path.basename(os.path.join(preprocessed_img, imageFile)))
    plt.show()

#Now that the images are thresholded I make 2D slices of every patient
#Slicing
for file in os.listdir(preprocessed_img):
    nii_img = nib.load(os.path.join(preprocessed_img, file))
    nii_msk = nib.load(os.path.join(maskPath, file))
    nii_data = nii_img.get_fdata()

    for i in range(nii_data.shape[2]):
        slice_img = nii_data[:, :, i]
        nii_msk_data = nii_msk.get_fdata()
        slice_msk = nii_msk_data[:, :, i]
        plt.imsave(os.path.join(img_slices, file[:-4] + "_" + str(i) + ".png"), slice_img, cmap='gray')
        plt.imsave(os.path.join(mask_slices, file[:-4] + "_" + str(i) + ".png"), slice_msk, cmap='gray')

#Now I will separate the images into different folders to train a better model.

#The goal is to train the IA with images with hemorrhages, that's why I have a folder to store the images with smaller or non-hemorhages and another to store the bigger hemorrhages

#So in order to do this first of all I am going to remove all the slices with a lesion smaller than 400 pixels
n = 0
threshold = 400
for filename in os.listdir(img_slices):
    # Open the label image and count the pixels with the value 1
    label_img = Image.open(os.path.join(mask_slices, filename)).convert('L')
    matriz_imagen = label_img.load()
    contador = 0
    for y in range(512):
        for x in range(512):
        # Get pixel value at position (x, y)
            valor_pixel = matriz_imagen[x, y]
            if valor_pixel > 100:
              contador += 1
    # If the image meets the threshold, copy it to the new folder
    if contador >= threshold:
        n += 1
        print(n)
        os.makedirs(slices_with_hemorrhages+"/label/label", exist_ok=True)
        os.makedirs(slices_with_hemorrhages+"/data/data", exist_ok=True)
        new_label_path = os.path.join(slices_with_hemorrhages+"/label/label", filename)
        new_data_path = os.path.join(slices_with_hemorrhages+"/data/data", filename)
        label_img.save(new_label_path)
        data_img = Image.open(os.path.join(img_slices, filename))
        data_img.save(new_data_path)

        
#Preprocessed Data Visualization
#Now we can visualize every slice preprocessed and its corresponding Groundtruth

# Set the number of images to process in each batch
batch_size = 2

# Find all .nii images in the images folder
imageFiles = os.listdir(slices_with_hemorrhages+"/data/data")

# Split the list of image files into batches of size batch_size
imageBatches = [imageFiles[i:i+batch_size] for i in range(0, len(imageFiles), batch_size)]

# Process each batch of images
for imageBatch in imageBatches:
    # Create a new figure with subplots for each image in the batch
    fig, axs = plt.subplots(nrows=len(imageBatch), ncols=2, figsize=(7,7))

    # Process each image in the batch
    for i, imageFile in enumerate(imageBatch):
        #Check if there is a file with the same name in the maskPath folder
        maskFile = os.path.join(slices_with_hemorrhages+"/label/label", imageFile)
        if os.path.exists(maskFile):
            # If there is a file with the same name in the maskPath folder, load both images and display them
            img = Image.open(os.path.join(img_slices, imageFile))
            axs[i, 0].imshow(img, cmap='gray')
            axs[i, 0].set_title(os.path.basename(os.path.join(img_slices, imageFile)))

            msk = Image.open(os.path.join(mask_slices, maskFile))
            axs[i, 1].imshow(msk, cmap='gray')
            axs[i, 1].set_title(os.path.basename(maskFile))

    # Show the figure
    plt.show()

#Now I 'll save the folder with all the preprocessed images, not just the ones with hemorrhages inside another folder ready to use on CNN

# In order to reorganize the data I copy the images from img_slice to all_data_preprocessed/data/data

for filename in os.listdir(img_slices):
    src_path = os.path.join(img_slices, filename)
    dst_path = os.path.join(all_img_prep, filename)
    shutil.copyfile(src_path, dst_path)
    
# In order to reorganize the data I copy the images from img_slice to all_data_preprocessed/label/label
for filename in os.listdir(mask_slices):
    src_path = os.path.join(mask_slices, filename)
    dst_path = os.path.join(all_mask_org, filename)
    shutil.copyfile(src_path, dst_path)

#Iuse this function to recount the images in each folder, it sometimes happens that the images doesn't copy due to connection problems or Google server problems
def count_images_in_folder(folder_path):
    """
        Counts the number of valid images in the specified folder.

   Args:
       folder_path (str): Path to the folder containing the images.

   Returns:
        int: Number images found in the folder.
"""
    count = 0
    for filename in os.listdir(folder_path):
        try:
            Image.open(os.path.join(folder_path, filename))
            count += 1
        except:
            pass
    return count


count_images_in_folder(img_slices_hemo)
count_images_in_folder(all_img_prep)


#Now I organize all the images in patients folders
# source folder path
source_folder = all_img_prep

# iterate through each file in the source folder
for filename in os.listdir(source_folder):
    
   # extract the patient number and slice number from the file name
    patient_num, slice_num = filename.split('_')
    patient_folder = os.path.join(patient_img, patient_num)

    # if the folder for the patient does not already exist, create it
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

    # copy the file to the corresponding patient folder
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(patient_folder, filename)
    shutil.copy(source_file, destination_file)

# source folder path
source_folder = all_mask_org

# iterate through each file in the source folder
for filename in os.listdir(source_folder):
    
# extract the patient and slice numbers from the file name
    patient_num, slice_num = filename.split('_')
    patient_folder = os.path.join(patient_mask, patient_num)

    # if the folder for the patient does not already exist, create it
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)

# copy the file to the corresponding patient folder
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(patient_folder, filename)
    shutil.copy(source_file, destination_file)



 # Iterate through each patient folder
for patient_folder in os.listdir(patient_img):
    # Determine whether to copy the patient folder to the train or test folder(80% to train and 20% to test)
    if random.random() < 0.8:
        destination_folder = os.path.join(patient_img_train, patient_folder)
        destination_label_folder = os.path.join(patient_mask_train, patient_folder)
    else:
        destination_folder = os.path.join(patient_img_test, patient_folder)
        destination_label_folder = os.path.join(patient_mask_test, patient_folder)

    # Copy the patient folder and its contents to the appropriate destination
    source_folder = os.path.join(patient_img, patient_folder)
    shutil.copytree(source_folder, destination_folder)
    
    # Copy the corresponding label folder to the appropriate destination if it exists
    source_label_folder = os.path.join(patient_mask, patient_folder)
    if os.path.exists(source_label_folder):
        shutil.copytree(source_label_folder, destination_label_folder)


directory = patient_img_train

# Get a list of all items in the directory
items = os.listdir(directory)

# Count the number of folders
num_folders = len([item for item in items if os.path.isdir(os.path.join(directory, item))])

# Print the number of folders
print(f"The directory '{directory}' contains {num_folders} folders.")


# Get a list of all files in the images folder
files = os.listdir(img_slices_hemo)

# Shuffle files randomly
random.shuffle(files)

# Calculate the number of files for the division in train and test
num_train = int(0.8 * len(files))
num_test = len(files) - num_train

print(num_train)#78
print(num_test)#17


# Copy the image and skin files to the corresponding folders
for i, file in enumerate(files):
  print(i)
  if os.path.exists(os.path.join(img_slices_hemo, file) and os.path.join(mask_slices_hemo, file)):
      if i < num_train:
          # Copia los archivos de train a las carpetas train_img y train_mask
          shutil.copy(os.path.join(img_slices_hemo, file), os.path.join(train_img, file))
          shutil.copy(os.path.join(mask_slices_hemo, file), os.path.join(train_mask, file))
      else:
          # Copia los archivos de test a las carpetas test_img y test_mask
          shutil.copy(os.path.join(img_slices_hemo, file), os.path.join(test_img, file))
          shutil.copy(os.path.join(mask_slices_hemo, file), os.path.join(test_mask, file))



def copy_images(source_folder, destination_folder):
    """
    Copies image files from the source folder to the destination folder.

    Args:
        source_folder (str): Path to the folder containing the source images.
        destination_folder (str): Path to the folder where the images will be copied.

    Returns:
        None
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all the files in the source folder
    files = os.listdir(source_folder)

    # Iterate over the files and copy the images
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)

        # Check if the file is an image
        if os.path.isfile(source_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {file}")

    print("Image copying completed.")

copy_images(os.path.join(results, 'train/label/label'), os.path.join(TFG, 'Last training/label/label'))

#I had problems with the image names, so i used this function to rectify the names
def extract_digits_from_images(folder_path):
    """
    Extracts the unique first three digits from image filenames in the specified folder.

    Args:
        folder_path (str): Path to the folder containing the images.

    Returns:
        set: Set of unique first three digits extracted from the filenames.
    """
    unique_digits = set()

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_path)

    # Iterar sobre los archivos
    for file in files:
        # Verificar si el archivo es una imagen
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Extraer los primeros tres dígitos del nombre de archivo
            digits = file.split('_')[0]

            # Agregar los dígitos a la lista de únicos
            unique_digits.add(digits)

    return unique_digits

unique_digits = extract_digits_from_images(os.path.join(TFG, 'Last training/data/data'))
# Imprimir el número de elementos en la lista
print("Número de elementos en la lista:", len(unique_digits))


#Since at this point I didn't take into account that I shouldn't have slices of the same patient in train and test, I had to remove the images from here
def extract_last_20(folder_path):
    """
    Extracts the last 20% of image filenames in the specified folder based on their numeric prefixes.

    Args:
        folder_path (str): Path to the folder containing the images.

    Returns:
        list: List of numeric prefixes of the removed images.
    """
    unique_digits = []

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_path)

    # Iterar sobre los archivos
    for file in files:
        # Verificar si el archivo es una imagen
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Extraer los primeros tres dígitos del nombre de archivo
            digits = int(file.split('_')[0])

            # Agregar los dígitos a la lista de únicos
            unique_digits.append(digits)

    # Ordenar la lista de números de menor a mayor
    unique_digits.sort()

    # Determinar el número de elementos a eliminar
    num_elements_to_remove = int(len(unique_digits) * 0.2)

    # Obtener los números de las imágenes que serán eliminadas
    removed_digits = unique_digits[-num_elements_to_remove:]

    # Eliminar las imágenes correspondientes
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            digits = int(file.split('_')[0])
            if digits in removed_digits:
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)

    return removed_digits

folder_path = os.path.join(TFG, 'Last training/data/data')

removed_digits = extract_last_20(folder_path)

# Print the number of the deleted images
print("Números de las imágenes eliminadas:", removed_digits)

from collections import OrderedDict
print("Números de las imágenes eliminadas:", list(OrderedDict.fromkeys(removed_digits)))


def copy_folders_with_numbers(source_dir, destination_dir, numbers):
    # Get list of folders in source directory

    # Copy the folders to the destination directory
    for n in numbers:
        source_path = os.path.join(source_dir, '0' + str(n))
        destination_path1 = os.path.join(destination_dir,'0' + str(n))
        destination_path = os.path.join(destination_path1,'0' + str(n))


        # Copy the folder and its contents
        shutil.copytree(source_path, destination_path)

        print(f"Copiada la carpeta: {n}")

    print("Copia de carpetas completada.")


source_dir = os.path.join(results, "Patient Slices/general/data/data")
destination_dir = os.path.join(TFG, "Last testing/data")
numbers = [82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 96, 97, 98, 99]
#The number 100 was addressed manually, even though it could have been done here
copy_folders_with_numbers(source_dir, destination_dir, numbers)
