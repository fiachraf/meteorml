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

# )
data_dir = input("directory that contains solely Confirmed and Rejected folders, labelled as 1 and 0 respectively: ")


train_data = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    interpolation="bilinear",
)
image_paths = train_data.file_paths
print(f"type(image_paths): {type(image_paths)}")
print(f"type(train_data): {type(train_data)}")
# print(f"train_data.shape(): {train_data.shape()}")

#images_batch and labels_batch are tensorflow.python.framework.ops.EagerTensor, they have a len of batch_size as defined in the keras.preprocessing.image_dataset_from_directory() function. Every time you get to the end of one batch, it gets replaced by the nex batch and so indexing can be a little tricky
#image_paths is a python list that contains all the full file paths of the images found by the keras.preprocessing.image_dataset_from_directory() and does not have any batches and so to pair up the file names and the actual entries in the tensors, you need to keep track of different indices
index = 0
for images_batch, labels_batch in train_data:
    print(f"len(images_batch): {len(images_batch)}")
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
        print(f"image_paths[{index}]: {image_paths[index]}")
        index += 1
        plt.imshow(image)
        plt.show()
    # print(f"type(x): {type(x)}")
    # print(f"type(y): {type(y)}")
    # print(f"len(train_data): {len(train_data)}")
    # print(f"x : {x}")
