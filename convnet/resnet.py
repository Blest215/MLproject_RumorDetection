"""
ResNet
"""

import tensorflow as tf
import functools

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


@add_arg_scope
def max_pool1d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None):
  """Adds a 1D Max Pooling op."""
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'MaxPool1D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = pooling_layers.MaxPooling1D(pool_size=kernel_size,
                                        strides=stride,
                                        padding=padding,
                                        data_format=df,
                                        _scope=sc)
    outputs = layer.apply(inputs)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


def conv1d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 1-D convolution with 'SAME' padding."""
  if stride == 1:
    return layers.convolution(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='SAME',
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = array_ops.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return layers.convolution(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        scope=scope)


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions."""
  if factor == 1:
    return inputs
  else:
    return max_pool1d(inputs, 1, stride=factor, scope=scope)


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model. (deprecated)
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'is_training': is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
  }

  with arg_scope(
      [layers.convolution],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm):
    with arg_scope([layers.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
      with arg_scope([max_pool1d], padding='SAME') as arg_sc:
        return arg_sc



@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN before convolutions."""
  with variable_scope.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    try:
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=3)
    except ValueError:
        depth_in = 5000
    preact = layers.batch_norm(
        inputs, activation_fn=nn_ops.relu, scope='preact')
    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.convolution(
          preact,
          depth, 1,
          stride=stride,
          normalizer_fn=None,
          activation_fn=None,
          scope='shortcut')

    residual = layers.convolution(
        preact, depth_bottleneck, 1, stride=1, scope='conv1')
    residual = conv1d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = layers.convolution(
        residual,
        depth, 1,
        stride=1,
        normalizer_fn=None,
        activation_fn=None,
        scope='conv3')

    output = shortcut + residual

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v2_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v2 bottleneck block."""
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=None,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  with variable_scope.variable_scope(
      scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with arg_scope(
        [layers.convolution, bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      if is_training is not None:
        bn_scope = arg_scope([layers.batch_norm], is_training=is_training)
      else:
        bn_scope = arg_scope([])
      with bn_scope:
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with arg_scope(
              [layers.convolution], activation_fn=None, normalizer_fn=None):
            net = conv1d_same(net, 64, 7, stride=2, scope='conv1')
          net = max_pool1d(net, 3, stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = layers.batch_norm(
            net, activation_fn=nn_ops.relu, scope='postnorm')
        if global_pool:
          # Global average pooling.
          net = math_ops.reduce_mean(net, [1], name='pool5', keep_dims=True)
        if num_classes is not None:
          net = layers.convolution(
              net,
              num_classes, 1,
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
          net = tf.squeeze(net)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          try:
            end_points['predictions'] = layers.softmax(net, scope='predictions')
          except ValueError:
            end_points['predictions'] = layers.softmax(tf.reshape(net, [1, num_classes]), scope='predictions')
        return net, end_points


def resnet(inputs,
           num_classes=None,
           is_training=True,
           global_pool=True,
           output_stride=None,
           reuse=None,
           scope='resnet'):
    blocks = {
        '50': [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
            resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ],
        '101': [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
            resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ],
        '152': [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
            resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ],
        '200': [
            resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
            resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
            resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]
    }
    # blocks = [
    #     resnet_v2_block('block1', base_depth=128, num_units=3, stride=2),
    #     resnet_v2_block('block2', base_depth=256, num_units=4, stride=2),
    #     resnet_v2_block('block3', base_depth=512, num_units=6, stride=1),
    # ]

    inputs = tf.to_float(inputs)
    return resnet_v2(
        inputs,
        blocks['152'],
        num_classes,
        is_training,
        global_pool,
        output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)
resnet.default_image_size = 96


def get_resnet_func(num_classes, weight_decay=0.0, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    func = resnet
    @functools.wraps(func)
    def resnet_fn(images):
        arg_scope = resnet_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training)

    if hasattr(func, 'default_image_size'):
        resnet_fn.default_image_size = func.default_image_size

    return resnet_fn