#import tensorflow as tf
#from tf import keras
from tensorflow.keras import layers, models

#short script to view structure of model as I forgot to record exact details of each model

model_name = input("name of keras model")
CNN_model = models.load_model(model_name)
CNN_struct = CNN_model.get_config()
print(CNN_struct)
