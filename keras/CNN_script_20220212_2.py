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


#keras preprocessing, will resize images etc.
# tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(256, 256),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False,
#     **kwargs
# )
data_dir = input("directory that contains solely Confirmed and Rejected folders, labelled as 1 and 0 respectively: ")
output_model_name = input("model name that you want to give it e.g. meteorml_20220201 (do not include the .5): ")


#keras preprocessing, will resize images etc. create training and validation datasets
train_data = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=169,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
)


#need to create new datasets like this as it doesn't seem to work properly if the dataset is modified in place for some reason
train_norm = train_data.map(normalise_me)
train_cent = train_norm.map(center_me)
train_stan = train_cent.map(standardise_me)
train_resc2 = train_stan.map(rescale_2)


val_data = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="grayscale",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=169,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
)
val_norm = val_data.map(normalise_me)
val_cent = val_norm.map(center_me)
val_stan = val_cent.map(standardise_me)
val_resc2 = val_stan.map(rescale_2)

#sample netowrk, first layer has to include an argument for input shape
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dense(1, activation="sigmoid"))

#prints a summary of the model
#model.summary

#import matplotlib.pyplot as plt
#
#for image_1_batch, label_batch in train_cent:
#    print("test")
#    print(tf.shape(image_1_batch), tf.shape(image_2_batch), tf.shape(label_batch))
#    for image_1, image_2, label in zip(image_1_batch, image_2_batch, label_batch):
#        # print(tf.shape(image_1), tf.shape(image_2), tf.shape(label))
#        print(f"label: {label}")
#        plt.imshow(image_1)
#        plt.show()
#        plt.imshow(image_2)
#        plt.show()
#similar recreation of model used by Peter S. Gural as shown in Table 3 in his paper:
# Deep learning algorithms applied to the classification of video meteor detections, Peter S. Gural, doi:10.1093/mnras/stz2456
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(128,128,1)))
model.add(layers.MaxPooling2D((5,5)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((5,5)))
#model.add(layers.Conv2D(64, (3,3), activation="relu"))
#model.add(layers.MaxPooling2D((5,5)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))


model.summary()

#compile the network
from keras import optimizers

model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(), metrics=["acc"])



#fit the model to the data
#steps_per_epoch is the number of training steps the code runs before beginning a new epoch, exclude this line to run over the whole dataset for each epoch
history = model.fit(
train_resc2,
# steps_per_epoch=36,
epochs=10,
validation_data=val_resc2)
# validation_steps=36)

model.save(output_model_name + ".h5")


import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) +1)

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig("accuracy" + output_model_name + ".png")
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig("loss" + output_model_name + ".png")
plt.show()
