import PIL
from PIL import image
import os
import csv

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


model_name = input("enter the full path of model: ")
CCN_model = models.load_model(model_name)



folder_input = input("folder containing images that you want to test false identification: ")

#need to check which is confirmed and which is negative 0 or 1
#if ConfirmedFiles are labelled 1 then any predictions of 0 by the network by the files in this folder will be false negatives
if folder_input[-1] == 1:
    false_pos_or_neg = "false negative"
    #if RejectedFiles are labelled 0 then any predictions of 1 by the network by the files in this folder will be false positives
elif folder_input[-1] == 0:
    false_pos_or_neg = "false positive"
else:
    print("folder is set up wrong")
file_list = os.listdir(folder_input)

false_list = []
for item in file_list:
    prediction = model_name.predict(item)

    if prediction != folder_input[-1]:
        false_list.append(item)

log_file_name = false_iden_log.csv
with open(log_file_name, "a") as csv_logfile:
    csv_logfile_writer = csv.writer(csv_logfile, delimiter=",")
    for item in false_list:
        csv_logfile_writer.writerow([item])
