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
from custom_keras import image_dataset

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

data_dir_1 = input("directory that contains detection_pixel images (maxpixel - avgpixel), solely Confirmed and Rejected folders, labelled as 1 and 0 respectively: ")

data_dir_2 = input("directory containing maxframe images: ")

output_model_name = input("model name that you want to give it e.g. meteorml_20220201 (do not include the .5): ")


#keras preprocessing, will resize images etc. create training and validation datasets
train_data = image_dataset.cust_image_dataset_from_directory(
    data_dir_1,
    data_dir_2,
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

val_data = image_dataset.cust_image_dataset_from_directory(
    data_dir_1,
    data_dir_2,
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

#need to create new datasets like this as it doesn't seem to work properly if the dataset is modified in place for some reason
train_norm = train_data.map(image_dataset.normalise_me)
train_cent = train_norm.map(image_dataset.center_me)
train_stan = train_cent.map(image_dataset.standardise_me)
val_norm = val_data.map(image_dataset.normalise_me)
val_cent = val_norm.map(image_dataset.center_me)
val_stan = val_cent.map(image_dataset.standardise_me)


#this creates a Sequential model, which can't handle multiple inputs
"""
----------------------------------------------------------------------------
#similar recreation of model used by Peter S. Gural as shown in Table 3 in his paper:
# Deep learning algorithms applied to the classification of video meteor detections, Peter S. Gural, doi:10.1093/mnras/stz2456
model = models.Sequential()
#model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))



------------------------------------------------------------------------------
"""
#sample of sequential model and its functional equivalent from Francois Chollet pg.237
"""
#the sequential model:
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

#the functional model
#each layer is linked together by calling each function on the previous layer, e.g. first x = has (input_tensor) at end of line and then each x= line has (x) at the end, output_tensor then has (x) at the end of the line too
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
#Model class converts, input tensor and output tensor to model
"""
#need to use keras functional api style to create model
#input = Input(shape=(None,), dtype='int32', name='text') #dtype can be manually set and input can also be given a name e.g. 'text'
#inputs has properties .shape and .dtype
# input_tensor = keras.Input(shape=(128,128,1))
detectpixel_input =keras.Input(shape=(128,128,1))
maxframe_input = keras.Input(shape=(128,128,1))
#framework for the detectpixel_input
x_frame = layers.Conv2D(64, (3,3), activation="relu")(detectpixel_input)
x_frame = layers.MaxPooling2D((2,2))(x_frame)
x_frame = layers.Conv2D(64, (3,3), activation="relu")(x_frame)
x_frame = layers.MaxPooling2D((2,2))(x_frame)
#x_frame = layers.Conv2D(64, (3,3), activation="relu")(x_frame)
#x_frame = layers.MaxPooling2D((2,2))(x_frame)
x_frame = layers.Flatten()(x_frame)
x_frame = layers.Dense(64, activation="relu")(x_frame)
#x_frame = layers.Dense(64, activation="sigmoid")(x_frame)


#framework for the maxframe input
y_frame = layers.Conv2D(64, (3,3), activation="relu")(maxframe_input)
y_frame = layers.MaxPooling2D((2,2))(y_frame)
y_frame = layers.Conv2D(64, (3,3), activation="relu")(y_frame)
y_frame = layers.MaxPooling2D((2,2))(y_frame)
#y_frame = layers.Conv2D(64, (3,3), activation="relu")(y_frame)
#y_frame = layers.MaxPooling2D((2,2))(y_frame)
y_frame = layers.Flatten()(y_frame)
y_frame = layers.Dense(64, activation="relu")(y_frame)
#x_frame = layers.Dense(64, activation="sigmoid")(x_frame)

#need to combine the two frameworks
concat_layer = layers.Concatenate()([x_frame, y_frame])

combo_layer = layers.Dense(256, activation="relu")(concat_layer)
#combo_layer = layers.Dense(64, activation="relu")(combo_layer)
output = layers.Dense(1, activation="sigmoid")(combo_layer)

model = models.Model([detectpixel_input, maxframe_input], outputs=output)


#functional api uses same syntax as sequential style for compiling, evaluating, training

model.summary()
#requires pydot to be installed
#keras.utils.plot_model(model, "multi_input_keras.png", show_shapes=True)
#compile the network
from keras import optimizers

model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(epsilon=0.1), metrics=["acc"])

#fit the model to the data
#steps_per_epoch is the number of training steps the code runs before beginning a new epoch, exclude this line to run over the whole dataset for each epoch
history = model.fit(
train_stan,
# steps_per_epoch=100,
epochs=10,
validation_data=val_stan)
#validation_steps=50)


#unsure which set to call the validation set and which to call the testing set. It seems several sources use them to refer to the wrong sets but they are two different sets. One is used while training the network so that it can determine its performance and the other is used to provide a final value for the network's performance when it is in its final form/iteration

#sample code
# history is a dictionary that contains items for each value being tracked. E.g loss and accuracy values. If the loss and accuracy of the network is being tracked for both the training and testing/validation data is being tracked, history will contain 4 values; accuracy for training data, loss for training data, accuracy for testing/validation data and loss for testing/validation data
# history = model.fit(partial_x_train,  #partial_x_train is a tensor containing the training data
# partial_y_train,  #partial_y_train is a list/tensor containing labels for the training data
# epochs=20,    #number of iterations of training over all samples in the dataset
# batch_size=512,   #number of samples to be used in one go to perform mini-batch stochastic gradient descent operation
# validation_data=(x_val, y_val))   #validation data and validation labels as one tuple

#plotting the performance of the network on a graph, first graph is the plot of the loss_values, second graph is a plot of the accuracy
# import matplotlib.pyplot as plt
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(acc) + 1)
# “bo” is for “blue dot.” and “b” is for “solid blue line.”
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.clf()
# Clears the figure
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# can also use next style instead when not using list/tensors that are already in memory, and are instead using generators which load read the data in batches
# history = model.fit_generator(
# train_generator,  #the data set to be trained on
# steps_per_epoch=100,
# epochs=30,
# validation_data=validation_generator,
# validation_steps=50)




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
