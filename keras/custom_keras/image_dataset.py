# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras image dataset loading utilities."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.util.tf_export import keras_export

import tensorflow as tf

ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')


@keras_export('keras.preprocessing.image_dataset_from_directory', v1=[])
def image_dataset_from_directory(directory,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False):
  """Generates a `tf.data.Dataset` from image files in a directory.
  If your directory structure is:
  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```
  Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.
  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        or a list/tuple of integer labels of the same size as the number of
        image files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    batch_size: Size of the batches of data. Default: 32.
    image_size: Size to resize images to after they are read from disk.
        Defaults to `(256, 256)`.
        Since the pipeline processes batches of images that must all have
        the same size, this must be provided.
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    interpolation: String, the interpolation method used when resizing images.
      Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.
  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.
  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.
  """
  if labels != 'inferred':
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          '(no labels), pass `label_mode=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
        'Received: %s' % (color_mode,))
  interpolation = image_preprocessing.get_interpolation(interpolation)
  dataset_utils.check_validation_split_arg(
      validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
  image_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

  image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)

  dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation)
  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = image_paths
  return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  img_ds = path_ds.map(
      lambda x: path_to_image(x, image_size, num_channels, interpolation))
  if label_mode:
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
  return img_ds


def path_to_image(path, image_size, num_channels, interpolation):
  img = io_ops.read_file(path)
  img = image_ops.decode_image(
      img, channels=num_channels, expand_animations=False)
  img = image_ops.resize_images_v2(img, image_size, method=interpolation)
  img.set_shape((image_size[0], image_size[1], num_channels))
  return img

###################################
#custom functions


@keras_export('keras.preprocessing.image_dataset_from_directory', v1=[])
def cust_image_dataset_from_directory(directory_1,
                                 directory_2,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False):
  """Generates a `tf.data.Dataset` from image files in a directory.
  If your directory structure is:
  ```
  main_directory/
  ...class_a/
  ......a_image_1.jpg
  ......a_image_2.jpg
  ...class_b/
  ......b_image_1.jpg
  ......b_image_2.jpg
  ```
  Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
  will return a `tf.data.Dataset` that yields batches of images from
  the subdirectories `class_a` and `class_b`, together with labels
  0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
  Supported image formats: jpeg, png, bmp, gif.
  Animated gifs are truncated to the first frame.
  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    labels: Either "inferred"
        (labels are generated from the directory structure),
        or a list/tuple of integer labels of the same size as the number of
        image files found in the directory. Labels should be sorted according
        to the alphanumeric order of the image file paths
        (obtained via `os.walk(directory)` in Python).
    label_mode:
        - 'int': means that the labels are encoded as integers
            (e.g. for `sparse_categorical_crossentropy` loss).
        - 'categorical' means that the labels are
            encoded as a categorical vector
            (e.g. for `categorical_crossentropy` loss).
        - 'binary' means that the labels (there can be only 2)
            are encoded as `float32` scalars with values 0 or 1
            (e.g. for `binary_crossentropy`).
        - None (no labels).
    class_names: Only valid if "labels" is "inferred". This is the explict
        list of class names (must match names of subdirectories). Used
        to control the order of the classes
        (otherwise alphanumerical order is used).
    color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
        Whether the images will be converted to
        have 1, 3, or 4 channels.
    batch_size: Size of the batches of data. Default: 32.
    image_size: Size to resize images to after they are read from disk.
        Defaults to `(256, 256)`.
        Since the pipeline processes batches of images that must all have
        the same size, this must be provided.
    shuffle: Whether to shuffle the data. Default: True.
        If set to False, sorts the data in alphanumeric order.
    seed: Optional random seed for shuffling and transformations.
    validation_split: Optional float between 0 and 1,
        fraction of data to reserve for validation.
    subset: One of "training" or "validation".
        Only used if `validation_split` is set.
    interpolation: String, the interpolation method used when resizing images.
      Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    follow_links: Whether to visits subdirectories pointed to by symlinks.
        Defaults to False.
  Returns:
    A `tf.data.Dataset` object.
      - If `label_mode` is None, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
      - Otherwise, it yields a tuple `(images, labels)`, where `images`
        has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.
  Rules regarding labels format:
    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorial`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
  Rules regarding number of channels in the yielded images:
    - if `color_mode` is `grayscale`,
      there's 1 channel in the image tensors.
    - if `color_mode` is `rgb`,
      there are 3 channel in the image tensors.
    - if `color_mode` is `rgba`,
      there are 4 channel in the image tensors.
  """
  if labels != 'inferred':
    if not isinstance(labels, (list, tuple)):
      raise ValueError(
          '`labels` argument should be a list/tuple of integer labels, of '
          'the same size as the number of image files in the target '
          'directory. If you wish to infer the labels from the subdirectory '
          'names in the target directory, pass `labels="inferred"`. '
          'If you wish to get a dataset that only contains images '
          '(no labels), pass `label_mode=None`.')
    if class_names:
      raise ValueError('You can only pass `class_names` if the labels are '
                       'inferred from the subdirectory names in the target '
                       'directory (`labels="inferred"`).')
  if label_mode not in {'int', 'categorical', 'binary', None}:
    raise ValueError(
        '`label_mode` argument must be one of "int", "categorical", "binary", '
        'or None. Received: %s' % (label_mode,))
  if color_mode == 'rgb':
    num_channels = 3
  elif color_mode == 'rgba':
    num_channels = 4
  elif color_mode == 'grayscale':
    num_channels = 1
  else:
    raise ValueError(
        '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
        'Received: %s' % (color_mode,))
  interpolation = image_preprocessing.get_interpolation(interpolation)
  dataset_utils.check_validation_split_arg(
      validation_split, subset, shuffle, seed)

  if seed is None:
    seed = np.random.randint(1e6)
  image_paths_1, labels_1, class_names_1 = dataset_utils.index_directory(
      directory_1,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  image_paths_2, labels_2, class_names_2 = dataset_utils.index_directory(
      directory_2,
      labels,
      formats=ALLOWLIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

  if label_mode == 'binary' and len(class_names_1) != 2:
    raise ValueError(
        'When passing `label_mode="binary", there must exactly 2 classes. '
        'Found the following classes: %s' % (class_names,))

  image_paths_1, labels_1 = dataset_utils.get_training_or_validation_split(
      image_paths_1, labels_1, validation_split, subset)

  image_paths_2, labels_2 = dataset_utils.get_training_or_validation_split(
      image_paths_2, labels_2, validation_split, subset)

  dataset = cust_paths_and_labels_to_dataset(
      image_paths_1=image_paths_1,
      image_paths_2=image_paths_2,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels_1,
      label_mode=label_mode,
      num_classes=len(class_names_1),
      interpolation=interpolation)

  if shuffle:
    # Shuffle locally at each iteration
    dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
  dataset = dataset.batch(batch_size)
  # Users may need to reference `class_names`.
  dataset.class_names = class_names
  # Include file paths for images as attribute.
  dataset.file_paths = (image_paths_1, image_paths_2)
  return dataset



def cust_paths_and_labels_to_dataset(image_paths_1,
                                image_paths_2,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  path_ds_1 = dataset_ops.Dataset.from_tensor_slices(image_paths_1)
  img_ds_1 = path_ds_1.map(
      lambda x: path_to_image(x, image_size, num_channels, interpolation))
  path_ds_2 = dataset_ops.Dataset.from_tensor_slices(image_paths_2)
  img_ds_2 = path_ds_2.map(
      lambda x: path_to_image(x, image_size, num_channels, interpolation))

  if label_mode:
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds_1, img_ds_2, label_ds))
  else:
    img_ds = dataset_ops.Dataset.zip((img_ds_1, img_ds_2))
  return img_ds


#noramlisaton functions, can be done using keras preprocessinglayers which are present in newer versions of tensorflow, I am using tensorflow 2.4.1 I had a reson for this specific verions but I just can't remember
def normalise_me(image_1, image_2, label):
    image_1 = tf.cast(image_1/255., tf.float32)
    image_2 = tf.cast(image_2/255., tf.float32)
    return image_1, image_2, label

def center_me(image_1, image_2, label):
    #image_1 is really a batch of image_1s and so axis=[1,2,3] and keepdims=True is needed so that it gets the mean and standard deviation for each individula image_1 rather than across the batch
    mean_1 = tf.math.reduce_mean(image_1, axis=[1,2,3], keepdims=True)
    mean_2 = tf.math.reduce_mean(image_2, axis=[1,2,3], keepdims=True)
    print(f"mean_1: {mean_1}, mean_2: {mean_2}")
    image_1 = image_1 - mean_1
    image_2 = image_2 - mean_2
    return image_1, image_2, label

def standardise_me(image_1, image_2, label):
    std_val_1 = tf.math.reduce_std(image_1, axis=[1,2,3], keepdims=True)
    image_1 = image_2 / std_val_1
    std_val_2 = tf.math.reduce_std(image_2, axis=[1,2,3], keepdims=True)
    image_2 = image_2 / std_val_2
    return image_1, image_2, label


# image_paths_1, labels_1, class_names_1 = dataset_utils.index_directory(
#   directory="/home/fiachra/Downloads/Meteor_Files/20210201_pngs",
#   labels="inferred",
#   formats=ALLOWLIST_FORMATS,
#   class_names=None,
#   shuffle=False,
#   seed=None,
#   follow_links=False)
# image_paths_2, labels_2, class_names_2 = dataset_utils.index_directory(
#   directory="/home/fiachra/Downloads/Meteor_Files/20210201_pngs",
#   labels="inferred",
#   formats=ALLOWLIST_FORMATS,
#   class_names=None,
#   shuffle=False,
#   seed=None,
#   follow_links=False)
# test_ds_1 = paths_and_labels_to_dataset(image_paths_1,
#                                             (128,128),
#                                             num_channels=1,
#                                             labels=None,
#                                             label_mode=None,
#                                             num_classes=None,
#                                             interpolation="bilinear"
#                                             )
# test_ds = cust_paths_and_labels_to_dataset(image_paths_1,
#                                             image_paths_2,
#                                             (128,128),
#                                             num_channels=1,
#                                             labels=labels_1,
#                                             label_mode="binary",
#                                             num_classes=2,
#                                             interpolation="bilinear"
#                                             )

# test_ds_2 = image_dataset_from_directory(
#     "/home/fiachra/Downloads/Meteor_Files/20210201_pngs",
#     labels="inferred",
#     label_mode="binary",
#     class_names=None,
#     color_mode="grayscale",
#     batch_size=32,
#     image_size=(128, 128),
#     shuffle=True,
#     seed=169,
#     validation_split=0.2,
#     subset="training",
#     interpolation="bilinear",
# )
import matplotlib.pyplot as plt
# for image_batch, label_batch in test_ds_2:
#     for image in image_batch:
#         print(tf.shape(image))
#         plt.imshow(image)
#         plt.show()
# for image in test_ds_1:
#     print(tf.shape(image))
#     plt.imshow(image)
#     plt.show()
# print(tf.shape(test_ds))
# for image_pair in test_ds:
#     print(tf.shape(image_pair))
#     for image in image_pair:
#         print(tf.shape(image))
#         plt.imshow(image)
#         plt.show()
# for image_1, image_2, label in test_ds:
#     print(tf.shape(image_1), tf.shape(image_2), tf.shape(label))
#     print(f"label: {label}")
#     plt.imshow(image_1)
#     plt.show()
#     plt.imshow(image_2)
#     plt.show()

test_ds_3 = cust_image_dataset_from_directory("//mnt/local/fiachra/meteor_images/files/20220121_pngs",
                                 "/mnt/local/fiachra/meteor_images/files/20220201_1_pngs",
                                 labels='inferred',
                                 label_mode='binary',
                                 class_names=None,
                                 color_mode='grayscale',
                                 batch_size=32,
                                 image_size=(128, 128),
                                 shuffle=False,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False)

test_ds_3_norm = test_ds_3.map(normalise_me)
test_ds_3_cent = test_ds_3_norm.map(center_me)
test_ds_3_stan = test_ds_3_cent.map(standardise_me)

# for image_1_batch, image_2_batch, label_batch in test_ds_3:
#     print(tf.shape(image_1_batch), tf.shape(image_2_batch), tf.shape(label_batch))
for image_1_batch, image_2_batch, label_batch in test_ds_3_cent:
    print("test")
    print(tf.shape(image_1_batch), tf.shape(image_2_batch), tf.shape(label_batch))
    for image_1, image_2, label in zip(image_1_batch, image_2_batch, label_batch):
        # print(tf.shape(image_1), tf.shape(image_2), tf.shape(label))
        print(f"label: {label}")
        plt.imshow(image_1)
        plt.show()
        plt.imshow(image_2)
        plt.show()
    # print(batch)
    # print(tf.shape(batch))
    # for triple in batch:
    #     print(tf.shape(triple))
    # for image_1, image_2, label in batch:
        # print(tf.shape(image_1), tf.shape(image_2), tf.shape(label))
