#ensures that tensorflow is used as the backend
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory


#noramlisaton functions, can be done using keras preprocessinglayers which are present in newer versions of tensorflow, I am using tensorflow 2.4.1 I had a reson for this specific verions but I just can't remember
def normalise_me(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

def center_me(image, label):
#image is really a batch of images and so axis=[1,2,3] and keepdims=True is needed so that it gets the mean and standard deviation for each individula image rather than across the batch
    mean_1 = tf.math.reduce_mean(image, axis=[1,2,3], keepdims=True)
    image = image - mean_1
    return image, label

def standardise_me(image, label):
    std_val = tf.math.reduce_std(image, axis=[1,2,3], keepdims=True)
    image = image / std_val
    return image, label

def rescale_2(image, label):
    max_val =tf.math.reduce_max(tf.math.abs(image), axis=[1,2,3], keepdims=True)
    image = image / max_val
    return image, label


def cust_predict(file_dir, model_path):

    CNN_model = models.load_model(model_path)

    # make a tf.dataset generator list
    file_dataset = image_dataset_from_directory(
       file_dir,
       labels="inferred",
       # label_mode=None,
       color_mode = "grayscale",
       batch_size=32,
       #don't forget to change image size depending on network
       image_size = (32,32),
       shuffle=False,
       interpolation = "bilinear")
    #have to apply the transofrmations like this, doesn't seem to work to modify the dataset in place
    file_dataset_norm = file_dataset.map(normalise_me)
    file_dataset_cent = file_dataset_norm.map(center_me)
    file_dataset_stan = file_dataset_cent.map(standardise_me)
    file_dataset_resc2 = file_dataset_stan.map(rescale_2)

    image_paths = file_dataset.file_paths
    prediction_list = []
    #index varaible used to keep track of file names as images are loaded in batches while the filenames is just one long list
    index = 0
    for images_batch, labels_batch in file_dataset_resc2:
        prediction = CNN_model.predict(images_batch)
        # print(f"batch {(index + 1)/ 32} of {len(image_paths) / 32}")
        for index_2, image in enumerate(images_batch):
            #need the [index_2][0] slice as the [index_2] slice is an EagerTensor and the [0] slice then gets the actual value out of the EagerTensor
            pred = round(prediction[index_2][0], 2)
            prediction_list.append((pred, image_paths[index]))
            index += 1

    return prediction_list

if __name__ == "__main__":
    if len(sys.argv) != 1:
        png_dir_1 = sys.argv[1]
        model_path_1 = sys.argv[2]
    else:
        png_dir_1 = input("enter path of png file directory to be used: ")
        model_path_1 = input("enter path of .h5 model to be used: ")

    print(cust_predict(png_dir_1, model_path_1))
