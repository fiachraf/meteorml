#https://studymachinelearning.com/model-quantization-methods-in-tensorflow-lite/

import glob
import numpy as np
import tensorflow as tf

dir = ""
keras_model = 'meteorml_20220325_1.h5'
model=tf.keras.models.load_model(keras_model)
print(model.summary())

#train_images = []

# This enables the converter to estimate a dynamic range for all the variable data.
def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files(dir + '1/*.png')
    for i in range(130):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.resize(image, [32,32])
        image = tf.cast(image / 255., tf.float16)
        image = tf.expand_dims(image, 0)
        print(image.shape)
        yield [image]
        
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_data_gen

#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_types = [tf.float16]
#converter.target_spec.supported_types = [tf.int8]
#converter.inference_input_type = tf.float16
#converter.inference_output_type = tf.uint8
converter.allow_custom_ops = True

tflite_model = converter.convert()

# Save the model.
with open('meteorml32.tflite', 'wb') as f:
  f.write(tflite_model)