from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from magenta.models.image_stylization import image_utils


def imagenet_inputs(batch_size, image_size):
  """Loads a batch of coco inputs.
  Used as a replacement for inception.image_processing.inputs in
  tensorflow/models in order to get around the use of hard-coded flags in the
  image_processing module.
  Args:
    batch_size: int, batch size.
    image_size: int. The images will be resized bilinearly to shape
        [image_size, image_size].
  Returns:
    4-D tensor of images of shape [batch_size, image_size, image_size, 3], with
    values in [0, 1].
  """
  with tf.name_scope('batch_processing'):
    dataset = tf.data.Dataset.list_files('/home/reiichiro/Pictures/contents/*')

    def _parse_function(filename):
      image_string = tf.read_file(filename)
      image = tf.image.decode_jpeg(image_string, channels=3)

      image = image_utils._aspect_preserving_resize(image, image_size + 2)
      image = image_utils._central_crop([image], image_size, image_size)[0]

      image.set_shape([image_size, image_size, 3])
      image = tf.to_float(image) / 255.0

      return image

    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    # Create saveable object from iterator.
    # saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

    # Save the iterator state by adding it to the saveable objects collection.
    # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

    return image, 0


def arbitrary_style_image_inputs(style_dataset_images,
                                        batch_size=None,
                                        image_size=None,
                                        center_crop=True,
                                        shuffle=True,
                                        augment_style_images=False,
                                        random_style_image_size=False,
                                        min_rand_image_size=128,
                                        max_rand_image_size=300):
  """Loads a batch of random style image given the path of images folder.
  This method does not return pre-compute Gram matrices for the images like
  style_image_inputs. But it can provide data augmentation. If
  augment_style_images is equal to True, then style images will randomly
  modified (eg. changes in brightness, hue or saturation) for data
  augmentation. If random_style_image_size is set to True then all images
  in one batch will be resized to a random size.
  Args:
    style_dataset_images: str, paths to the style images.
    batch_size: int. If provided, batches style images. Defaults to None.
    image_size: int. The images will be resized bilinearly so that the smallest
        side has size image_size. Defaults to None.
    center_crop: bool. If True, center-crops to [image_size, image_size].
        Defaults to False.
    shuffle: bool, whether to shuffle style files at random. Defaults to False.
    augment_style_images: bool. Wheather to augment style images or not.
    random_style_image_size: bool. If this value is True, then all the style
        images in one batch will be resized to a random size between
        min_rand_image_size and max_rand_image_size.
    min_rand_image_size: int. If random_style_image_size is True, this value
        specifies the minimum image size.
    max_rand_image_size: int. If random_style_image_size is True, this value
        specifies the maximum image size.
  Returns:
    4-D tensor of shape [1, ?, ?, 3] with values in [0, 1] for the style
    image (with random changes for data augmentation if
    augment_style_image_size is set to true), and 0-D tensor for the style
    label, 4-D tensor of shape [1, ?, ?, 3] with values in [0, 1] for the style
    image without random changes for data augmentation.
  Raises:
    ValueError: if center cropping is requested but no image size is provided,
        or if batch size is specified but center-cropping or
        augment-style-images is not requested,
        or if both augment-style-images and center-cropping are requested.
  """
  if center_crop and image_size is None:
    raise ValueError('center-cropping requires specifying the image size.')
  if center_crop and augment_style_images:
    raise ValueError(
      'When augment_style_images is true images will be randomly cropped.')
  if batch_size is not None and not center_crop and not augment_style_images:
    raise ValueError(
      'batching requires same image sizes (Set center-cropping or '
      'augment_style_images to true)')

  with tf.name_scope('style_image_processing'):
    dataset = tf.data.Dataset.list_files(style_dataset_images)

    def _parse_function(filename):
      image_string = tf.read_file(filename)
      image = tf.image.decode_jpeg(image_string, channels=3)
      image.set_shape([None, None, 3])
      if image_size is not None:
        image_channels = image.shape[2].value
        if augment_style_images:
          image_orig = image
          image = tf.image.random_brightness(image, max_delta=0.8)
          image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          image = tf.image.random_hue(image, max_delta=0.2)
          image = tf.image.random_flip_left_right(image)
          image = tf.image.random_flip_up_down(image)
          random_larger_image_size = random_ops.random_uniform(
            [],
            minval=image_size + 2,
            maxval=image_size + 200,
            dtype=dtypes.int32)
          image = image_utils._aspect_preserving_resize(image, random_larger_image_size)
          image = tf.random_crop(
            image, size=[image_size, image_size, image_channels])
          image.set_shape([image_size, image_size, image_channels])

          image_orig = image_utils._aspect_preserving_resize(image_orig, image_size + 2)
          image_orig = image_utils._central_crop([image_orig], image_size, image_size)[0]
          image_orig.set_shape([image_size, image_size, 3])
        elif center_crop:
          image = image_utils._aspect_preserving_resize(image, image_size + 2)
          image = image_utils._central_crop([image], image_size, image_size)[0]
          image.set_shape([image_size, image_size, image_channels])
          image_orig = image
        else:
          image = image_utils._aspect_preserving_resize(image, image_size)
          image_orig = image

      image = tf.to_float(image) / 255.0
      image_orig = tf.to_float(image_orig) / 255.0

      return image, image_orig

    dataset = dataset.map(_parse_function)
    if batch_size is None:
      batch_size = 1
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    if random_style_image_size:
      # Selects a random size for the style images and resizes all the images
      # in the batch to that size.
      def _resize_func(image_batch, image_orig_batch):
        image_batch = image_utils._aspect_preserving_resize(image_batch,
                                                            random_ops.random_uniform(
                                                              [],
                                                              minval=min_rand_image_size,
                                                              maxval=max_rand_image_size,
                                                              dtype=dtypes.int32))
        return image_batch, image_orig_batch

      dataset = dataset.map(_resize_func)
    iterator = dataset.make_one_shot_iterator()
    image, image_orig = iterator.get_next()

    # Create saveable object from iterator.
    # saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

    # Save the iterator state by adding it to the saveable objects collection.
    # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)

    return image, 0, image_orig