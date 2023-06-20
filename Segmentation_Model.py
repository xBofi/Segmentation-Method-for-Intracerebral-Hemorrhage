#Now I am going to continue with the model construction, all the libraries needed are in the preprocessing script.
#The original script was divided to help the correction

# Define constants
SEED = 909
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


data_dir_train_image = os.path.join(TFG, 'Last training/data')
data_dir_train_mask = os.path.join(TFG, 'Last training/label')

data_dir_test_image = os.path.join(TFG, 'Last testing/data')
data_dir_test_mask = os.path.join(TFG, 'Last testing/label')

NUM_TRAIN = 563
NUM_TEST = 519

NUM_OF_EPOCHS = 100


def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
      """
    Creates a segmentation generator for training, which yields batches of augmented images and masks.

    Args:
        img_path (str): Path to the directory containing the input images.
        msk_path (str): Path to the directory containing the corresponding masks.
        BATCH_SIZE (int): Batch size for the generator.

    Returns:
        zip: A zip object containing the augmented images and masks.
    """
    data_gen_args = dict(rescale=1./255
#                      ,featurewise_center=True,
#                      featurewise_std_normalization=True,
#                      rotation_range=90,
#                      width_shift_range=0.2,
#                      height_shift_range=0.2,
#                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)


def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    """
    Creates a segmentation generator for testing, which yields batches of images and masks without augmentation.

    Args:
        img_path (str): Path to the directory containing the input images.
        msk_path (str): Path to the directory containing the corresponding masks.
        BATCH_SIZE (int): Batch size for the generator.

    Returns:
        zip: A zip object containing the images and masks.
    """
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, shuffle = False)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, shuffle = False)
    return zip(img_generator, msk_generator)
  
    
train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)



def display(display_list):
      """
    Displays a list of images in a grid format.

    Args:
        display_list (list): List of images to be displayed.

    Returns:
        None
    """
    plt.figure(figsize=(15,15))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

      
def show_dataset(datagen, num=1):
    """
    Displays a specified number of images and masks from a given data generator.

    Args:
        datagen (Generator): Data generator yielding image-mask pairs.
        num (int): Number of images to display. Default is 1.

    Returns:
        None
    """
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])
        
         
show_dataset(train_generator, 2)



def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
     """
    Creates a U-Net model with the specified number of levels and parameters.

    Args:
        n_levels (int): Number of levels in the U-Net.
        initial_features (int): Number of initial features for the convolutional layers. Default is 32.
        n_blocks (int): Number of convolutional blocks per level. Default is 2.
        kernel_size (int): Kernel size for the convolutional layers. Default is 3.
        pooling_size (int): Pooling size for the max pooling layers. Default is 2.
        in_channels (int): Number of input channels. Default is 1.
        out_channels (int): Number of output channels. Default is 1.

    Returns:
        keras.Model: U-Net model.
    """
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')
  
  
  
# Calculate the number of steps per epoch for training and testing data
EPOCH_STEP_TRAIN = NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

# Create a U-Net model with 4 classes
model = unet(4)

# Compile the model with Adam optimizer, binary crossentropy loss, and accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()

# Train the model using a generator for training data
# Set the number of steps per epoch for training and testing data
# Set the validation data and the number of validation steps
# Set the number of epochs
model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=NUM_OF_EPOCHS)

# Save the trained model to a specified directory
model.save(os.path.join(TFG, 'Model/1'))

# Load the saved model
model = tf.keras.models.load_model(os.path.join(TFG, 'Model/1'))

# Set the directories for the test image and mask data
data_dir_test_image = os.path.join(TFG, 'Last testing/data')
data_dir_test_mask = os.path.join(TFG, 'Last testing/label')


#I Create a test generator for each patient
test_generator_patient082 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '082'), os.path.join(data_dir_test_mask, '082'), 1)
test_generator_patient083 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '083'), os.path.join(data_dir_test_mask, '083'), 1)
test_generator_patient084 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '084'), os.path.join(data_dir_test_mask, '084'), 1)
test_generator_patient085 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '085'), os.path.join(data_dir_test_mask, '085'), 1)
test_generator_patient086 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '086'), os.path.join(data_dir_test_mask, '086'), 1)
test_generator_patient087 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '087'), os.path.join(data_dir_test_mask, '087'), 1)
test_generator_patient088 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '088'), os.path.join(data_dir_test_mask, '088'), 1)
test_generator_patient089 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '089'), os.path.join(data_dir_test_mask, '089'), 1)
test_generator_patient090 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '090'), os.path.join(data_dir_test_mask, '090'), 1)
test_generator_patient092 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '092'), os.path.join(data_dir_test_mask, '092'), 1)
test_generator_patient093 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '093'), os.path.join(data_dir_test_mask, '093'), 1)
test_generator_patient095 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '095'), os.path.join(data_dir_test_mask, '095'), 1)
test_generator_patient096 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '096'), os.path.join(data_dir_test_mask, '096'), 1)
test_generator_patient097 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '097'), os.path.join(data_dir_test_mask, '097'), 1)
test_generator_patient098 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '098'), os.path.join(data_dir_test_mask, '098'), 1)
test_generator_patient099 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '099'), os.path.join(data_dir_test_mask, '099'), 1)
test_generator_patient100 = create_segmentation_generator_test(os.path.join(data_dir_test_image, '100'), os.path.join(data_dir_test_mask, '100'), 1)

def show_prediction2(datagen, num=1, name = None):
        """
    Displays predictions and calculates evaluation metrics for a specified number of images using a given data generator.

    Args:
        datagen (Generator): Data generator yielding image-mask pairs.
        num (int): Number of images to display and evaluate. Default is 1.
        name (str): Name of the patient. Default is None.

    Returns:
        numpy.ndarray: Tensor of predicted masks.
        float: Dice coefficient for the patient.
        float: Percentage of false positives.
    """
    tensor = np.empty((512, 512, num))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = model.predict(image)[0] > 0.2

        tensor[:, :, i] = pred_mask[:, :, 0]
        print(name, i)

        A = np.asarray(mask).astype(np.bool)
        B = np.asarray(pred_mask).astype(np.bool)

        # Calculate the true positive, false positive and false negative
        TP = TP + np.count_nonzero(A & B)
        FP = FP + np.count_nonzero(B & ~A)
        FN = FN + np.count_nonzero(A & ~B)
        TN = TN + np.count_nonzero(~A & ~B)

        display([image[0], mask[0], pred_mask])
        # Calculate the Dice coefficient
    d = (2.0 * TP) / (2.0 * TP + FP + FN)
    print(f"The dice of patient {name} is: {d}")
    print(f"Percentage of false positives is: {(TP + TN)/(512*512*num)*100}")
    
    return tensor, d, (FP)/(512*512*num)*100
  
  
 # Create empty lists to store image data and evaluation metrics
image_vector = []
nifti_img = []
true_mask = []
dice_coefficient = []
false_positives = []

# Load the true mask from the specified path and add it to the true_mask list
true_mask.append(nib.load(os.path.join(maskPath, '082.nii')))

# Call the show_prediction2 function to make a prediction and obtain the tensor, dice coefficient, and false positives
tensor, dice, FP = show_prediction2(test_generator_patient082, 33, '082')

# Add the tensor, dice coefficient, and false positives to their respective lists
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

#Now the same with every patient
true_mask.append(nib.load(os.path.join(maskPath, '083.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient083, 32, '083')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '084.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient084, 32, '084')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '085.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient085, 28, '085')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '086.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient086, 28, '086')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '087.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient087, 27, '087')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '088.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient088, 40, '088')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '089.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient089, 28, '089')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '090.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient090, 30, '090')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '092.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient092, 30, '092')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '093.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient093, 32, '093')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '095.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient095, 34, '095')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '096.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient096, 32, '096')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '097.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient097, 28, '097')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '098.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient098, 28, '098')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '099.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient099, 31, '099')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)

true_mask.append(nib.load(os.path.join(maskPath, '100.nii')))
tensor, dice, FP = show_prediction2(test_generator_patient100, 30, '100')
image_vector.append(tensor)
dice_coefficient.append(dice)
false_positives.append(FP)
# Define a list of classes
classes = [82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100]

# Loop over the image_vector list
for i in range(len(image_vector)):
    # Create a NIfTI image using the image_vector, affine, and header from the corresponding true_mask
    nifti_img.append(nib.Nifti1Image(image_vector[i], true_mask[i].affine, true_mask[i].header))

# Define the output directory for the NIfTI files
output_nii_file = os.path.join(TFG + '/Volums')

# Loop over the nifti_img list
for i in range(len(nifti_img)):
    # Convert the class integer to a string and prepend '0'
    class_str = '0' + str(classes[i])
    # Save the NIfTI image to disk using the class string as the filename
    nib.save(nifti_img[i], os.path.join(output_nii_file, class_str))
    
# Print the mean of the dice_coefficient list
print(np.mean(dice_coefficient))

# Print the dice_coefficient list
print(dice_coefficient)

# Print the mean of the false_positives list
print(np.mean(false_positives))

# Print the false_positives list
print(false_positives)

# Define folder paths
folder_path = os.path.join(TFG, 'Volums')
label_folder_path = os.path.join(TFG, 'data_2/label')

# Initialize variables for accumulation
acumulated_dice = 0
acumulated_coincidence = 0
num_files = 0

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has the .nii extension
    if filename.endswith('.nii'):
        # Create file paths for the label and volume files
        nii_file1 = os.path.join(label_folder_path, filename)
        nii_file2 = os.path.join(folder_path, filename)

        # Load the NIfTI images
        A = nib.load(nii_file1)
        B = nib.load(nii_file2)

        # Get the image data as 3D arrays
        A_data = A.get_fdata().astype(bool)
        B_data = B.get_fdata().astype(bool)

        # Calculate true positives, false positives, false negatives, and true negatives
        TP = np.count_nonzero(A_data & B_data)
        FP = np.count_nonzero(B_data & ~A_data)
        FN = np.count_nonzero(A_data & ~B_data)
        TN = np.count_nonzero(~A_data & ~B_data)

        # Calculate the Dice coefficient
        dice_coefficient = (2.0 * TP) / (2.0 * TP + FP + FN)
        acumulated_dice += dice_coefficient

        # Calculate the coincidence
        coincidence = (TP + TN) / (TP + TN + FP + FN)
        acumulated_coincidence += coincidence

        num_files += 1

        # Print the evaluation metrics for each file
        print(f"File: {filename}")
        print(f"Índice DICE: {dice_coefficient}")
        print(f"Percentage of coincidence: {coincidence}")
        print(f"True positive: {TP}")
        print(f"False positive: {FP}")
        print(f"False negative: {FN}")
        print(f"True negative: {TN}")
        print("---------------------")

# Check if any files were processed
if num_files > 0:
    # Calculate and print the average Dice coefficient and average percentage of coincidence
    print(f"Average Dice coefficient: {acumulated_dice / num_files}")
    print(f"Average percentage of coincidence: {acumulated_coincidence / num_files}")
else:
    # Print a message if no files were found in the folder
    print("No files found in the folder.")
