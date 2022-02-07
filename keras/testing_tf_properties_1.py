#ensures that tensorflow is used as the backend
import tensorflow as tf
#tf.compact.v1.keras.backend.set_session()
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

#from keras.backend.tensorflow_backend import set_session
#import  tf.keras.backend.tensorflow_backend as K
#taken from Francis Chollet - Deep Learning with Python, starting page 147
#import keras
from tensorflow import keras
from tensorflow.keras import layers, models, Input
# from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np

# import custom_image_dataset_from_directory as custom

#neeed to have label as second input argument and label as the second value to be returned for these 3 functions
def normalise_me(image):
    image = tf.cast(image/255., tf.float32)
    return image

def center_me(image):
    mean_val = tf.math.reduce_mean(image, axis=None)
    print(f"mean_val: {mean_val}")
    image = tf.cast(image - mean_val, tf.float32)
    return image

def standardise_me(image):
    std_val = tf.math.reduce_std(image, axis=None)
    image = tf.cast(image/std_val, tf.float32)
    return image



data_dir_1 = input("directory that contains solely Confirmed and Rejected folders, labelled as 1 and 0 respectively: ")
# data_dir_2 = input("directory_2")

train_data_1 = keras.preprocessing.image_dataset_from_directory(
    data_dir_1,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    interpolation="bilinear",
)

# train_data_2 = keras.preprocessing.image_dataset_from_directory(
#     data_dir_2,
#     labels="inferred",
#     label_mode="binary",
#     class_names=None,
#     color_mode="grayscale",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=False,
#     interpolation="bilinear",
# )

# train_data_1_images, train_data_1_labels = tf.data.Dataset.unzip(train_data)
# print(f"tf.shape(train_data_1): {tf.shape(train_data_1)}")
# print(f"tf.shape(train_data_2): {tf.shape(train_data_2)}")
# print(f"tf.shape(train_data_combo): {tf.shape(train_data_combo)}")

# train_data_combo = tf.data.Dataset.zip((train_data_1, train_data_2))
# image_paths = train_data.file_paths
# print(f"type(image_paths): {type(image_paths)}")
# print(f"type(train_data): {type(train_data)}")

# layer = layers.LayerNormalization()
# train_data = layer(train_data)
# train_data_1 = train_data_1.map(normalise_me)
# train_data_1 = train_data_1.map(center_me)
# train_data_1 = train_data_1.map(standardise_me)

# print(f"train_data.shape(): {train_data.shape()}")

# rescale_layer = layers.Rescaling(1./255)
# normalise_layer = layers.LayerNormalization(axis=None)

#images_batch and labels_batch are tensorflow.python.framework.ops.EagerTensor, they have a len of batch_size as defined in the keras.preprocessing.image_dataset_from_directory() function. Every time you get to the end of one batch, it gets replaced by the nex batch and so indexing can be a little tricky
#image_paths is a python list that contains all the full file paths of the images found by the keras.preprocessing.image_dataset_from_directory() and does not have any batches and so to pair up the file names and the actual entries in the tensors, you need to keep track of different indices
index = 0
for images_batch, labels_batch in train_data_1:

    print(f"tf.shape(images_batch): {tf.shape(images_batch)}")

    # print(f"len(images_batch): {len(images_batch)}")
    #could do line to take slice from image_paths which has same length as images_batch and then remove these from the original list so that things can be done in batches rather than singly


    for image in images_batch:
        #Each image is a tensorflow.python.framework.ops.EagerTensor
        # numpy_array = image.numpy()   #tensorflow function to convert EagerTensor to numpy array
        # print(f"type(numpy_array): {type(numpy_array)}")
        # print(f"type(labels_batch): {type(labels_batch)}")
        # for idk in image:
        #     print(idk)
        #     plt.imshow(idk)
        # print(f"type(image): {type(image)}")
        # print(numpy_array)
        # print(f"image_paths[{index}]: {image_paths[index]}")
        index += 1
        # plt.imshow(image)
        # plt.show()
        output_1 = normalise_me(image)
        # plt.imshow(output_1)
        # plt.show()
        output_2 = center_me(output_1)
        plt.imshow(output_2)
        plt.show()
        # output_3 = standardise_me(output_2)

        # #this bit is only needed to plot the image for visual demonstrations
        # #-------------------------------------------------------------------
        # #create plot of meteor_image and crop_image
        # # has full image displayed on left side and then the can do two cropped images displayed on the right side
        # fig, axd = plt.subplot_mosaic([['og_image', 'normalise_image', 'center_image', 'standardise_image']])
        #
        # axd['og_image'].imshow(image)
        # axd['normalise_image'].imshow(output_1)
        # axd['center_image'].imshow(output_2)
        # axd['standardise_image'].imshow(output_3)
        #
        # # Create a Rectangle patch Rectangle((left column, top row), column width, row height, linewidth, edgecolor, facecolor)
        # rect = patches.Rectangle((left_side, top_side), (right_side - left_side), (bottom_side - top_side), linewidth=1, edgecolor='r', facecolor='none')
        #
        # # Add the patch to the big image
        # axd['big_image'].add_patch(rect)
        #
        # change the size of the figure
        # fig.set_size_inches(18.5, 10.5)
        #
        # display the plot
        # plt.show()
        #
        # #-------------------------------------------------------------------


    # print(f"type(x): {type(x)}")
    # print(f"type(y): {type(y)}")
    # print(f"len(train_data): {len(train_data)}")
    # print(f"x : {x}")
