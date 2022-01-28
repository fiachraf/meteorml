import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import time
from os.path import dirname
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
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
model_name = input("enter the full path of model: ")
CNN_model = models.load_model(model_name)

initial_dir = os.getcwd()

folder_input = input("folder containing folders labelled 0 and 1 respectively that contain images that you want to test false identification: ")
start_time = time.perf_counter()
#need to check which is confirmed and which is negative 0 or 1
#initialise list of predictions, elements are tuples (Label, prediction, false_pred, file_name)
#Label = 1 for Confirmed files, 0 for Rejected files, prediction = prediction value from neural network has value between 0 and 1, false_pred = Boolean true or false if the prediction is < 55% for Confirmed images or >45% for Rejected images it is considered false, file_name is the name of the .png file
# Confidence of less than 55% from prediction is considered false as it may be interesting to see which predictions the network is very unsure about
pred_list = []

def normalise(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

#make a tf.dataset generator list
#data_dir = folder_input
#file_dataset = image_dataset_from_directory(
#    data_dir,
#    labels="inferred",
#    color_mode = "grayscale",
#    batch_size=32,
#    image_size = (128,128),
#    interpolation = "bilinear")
#image_paths = file_dataset.file_paths
#labels_list = []
#for x, y in file_dataset:
#    labels_list.append(y)
#file_dataset = file_dataset.map(normalise)

#prediction_list = CNN_model.predict(file_dataset)
#iterate over all the files, first in the rejected files folder (0) and then in the Confirmed Files folder (1)

for i in range(2):
    os.chdir(folder_input + "/" + str(i))
    #print(f"cwd: {os.getcwd()}")
    file_list = os.listdir(folder_input + "/" + str(i))
    
    #print(f"file_list: {file_list}")
    for index, item in enumerate(file_list):
        #load image and convert to numpy array
        pil_image = load_img(item, color_mode = "grayscale", target_size = (128, 128), interpolation = "bilinear")
        image_np_array = img_to_array(pil_image)
        image_rescale = image_np_array/255.
        #converts to 4D tensor containing 1 image so that it works as an input to be predicted by the model as the model requires a 4d tensor for some reason
        tensor_4D = np.expand_dims(image_rescale, axis=0)
        prediction = CNN_model.predict(tensor_4D)
        #print(f"prediction: {prediction}")
        #print(f"type(prediction): {type(prediction)}")
        #print(f"prediction: {prediction}")
        pred = round(prediction[0][0], 2)
        if abs(i - pred) < 0.45:
            pred_correct = True
        else:
            pred_correct = False
        pred_list.append((i, round(pred,2), pred_correct, item))
        #print(f"file: {item}, prediction: {prediction}")
    

os.chdir(initial_dir)


def get_label(file_name):
    label_1_or_0 = int(dirname(file_name)[-1])
    return label_1_or_0

log_file_name = f"{model_name[:-3]}_false_iden_log_singly.csv"
with open(log_file_name, "a") as csv_logfile:
    csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
    csv_logfile_writer.writerow(["Label", "Prediction", "False Prediction", "File Name"])
    for index, pred in enumerate(pred_list):
        csv_logfile_writer.writerow([pred[0], pred[1], pred[2], pred[3]])

#pred_distribution_list = []
#for j in pred_list:
#    pred_distribution_list.append(i[1])

#need as input for hist function to be list of all pred values in pred_list
fig = plt.hist(pred_list, bins=100)
plt.title("Prediction_distribution")
plt.xlabel("Prediction")
plt.ylabel("No. of predictions")
plt.savefig(f"{model_name}_pred_dist_singly.png")

end_time = time.perf_counter()
print(f"time elapsed: {end_time - start_time}")
