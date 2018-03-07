# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
  inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
  kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
         Should be a positive integer.
  data_format: The input format ('channels_last' or 'channels_first').

  Returns:
  A tensor with the same format as the input with the data either intact
  (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                  [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def batch_normalization(inputs, data_format, is_training):
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs

def separable_conv2d(inputs, filters, kernel_size, strides, data_format, is_training):
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  inputs = tf.nn.relu(inputs)

  inputs = tf.layers.separable_conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
    depthwise_initializer=tf.variance_scaling_initializer(),
    pointwise_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)

  inputs = batch_normalization(inputs, data_format, is_training)
  return inputs

def average_pooling2d(inputs, pool_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, pool_size, data_format)

  inputs = tf.layers.average_pooling2d(
    inputs=inputs, pool_size=pool_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'),
    data_format=data_format)
  return inputs

def convolution_cell(last_inputs, inputs, params):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  data_format = params['data_format']
  is_training = params['is_training']
  node1 = last_inputs
  node2 = inputs
  node3 = separable_conv2d(node2, 64, 3, 1, data_format, is_training) + \
    tf.identity(node2)
  node4 = separable_conv2d(node2, 64, 5, 1, data_format, is_training) + \
    tf.identity(node1)
  node5 = average_pooling2d(node1, 3, 1, data_format) + \
    separable_conv2d(node2, 64, 3, 1, data_format, is_training)
  node6 = separable_conv2d(node1, 64, 3, 1, data_format, is_training) + \
    average_pooling2d(node2, 3, 1, data_format)
  node7 = separable_conv2d(node2, 64, 3, 1, data_format, is_training) + \
    average_pooling2d(node1, 3, 1, data_format)

  output = tf.concat([node3, node4, node5, node6, node7], axis=1 if data_format == 'channels_first' else 3)
  output = tf.layers.conv2d(
    inputs=output, filters=64, kernel_size=1, strides=1,
    padding='SAME', use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)
  output = tf.nn.relu(batch_normalization(output, data_format, is_training))
  return inputs, output
  

def reduction_cell(last_inputs, inputs, params):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  data_format = params['data_format']
  is_training = params['is_training']
  node1 = last_inputs
  node2 = inputs
  node3 = separable_conv2d(node1, 64, 5, 2, data_format, is_training) + \
    average_pooling2d(node2, 3, 2, data_format)
  node4 = separable_conv2d(node2, 64, 3, 2, data_format, is_training) + \
    average_pooling2d(node2, 3, 2, data_format)
  node5 = average_pooling2d(node2, 64, 2, data_format) + \
    separable_conv2d(node2, 64, 3, 2, data_format, is_training)
  node6 = separable_conv2d(node5, 64, 5, 2, data_format, is_training) + \
    average_pooling2d(node2, 3, 2, data_format)
  node7 = separable_conv2d(node6, 64, 3, 2, data_format, is_training) + \ 
    separable_conv2d(node1, 64, 5, 2, data_format, is_training)
  output = tf.concat([node3, node4, node7], axis=1 if data_format == 'channels_first' else 3)
  output = tf.layers.conv2d(
    inputs=output, filters=64, kernel_size=1, strides=1,
    padding='SAME', use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)
  output = tf.nn.relu(batch_normalization(output, data_format, is_training))
  return inputs, output

def build_block(inputs, params):
  num_cells = params['num_cells']
  data_format = params['data_format']
  is_training = params['is_training']
  inputs = tf.layers.conv2d(
    inputs=inputs, filters=64, kernel_size=1, strides=1,
    padding='SAME', use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)
  inputs = tf.nn.relu(batch_normalization(inputs, data_format, is_training))
  # first convolution_cell
  last_inputs, inputs = convolution_cell(last_inputs=inputs, inputs=inputs, params=params)
  for _ in xrange(1, num_cells):
    last_inputs, inputs = convolution_cell(last_inputs=last_inputs, inputs=inputs, params=params)
  last_inputs, inputs = reduction_cell(last_inputs=last_inputs, inputs=inputs, params=params)
  return inputs

def build_model(num_blocks, num_cells, num_nodes, num_classes, data_format=None):
  """Generator for net.

  Args:
  num_blocks: A single integer for the number of blocks.
  num_cells: A single integer for the number of convolution cells.
  num_nodes: A single integer for the number of nodes.
  num_classes: The number of possible classes for image classification.
  data_format: The input format ('channels_last', 'channels_first', or None).
    If set to None, the format is dependent on whether a GPU is available.

  Returns:
  The model function that takes in `inputs` and `is_training` and
  returns the output tensor of the model.
  """

  if data_format is None:
    data_format = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    params = {
      'num_blocks': num_blocks,
      'num_cells': num_cells,
      'num_nodes': num_nodes,
      'data_format': data_format,
      'is_training': is_training,
    }

    inputs = build_block(inputs=inputs, params=params)
    inputs = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=16, strides=1, padding='VALID', data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, 256])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model
