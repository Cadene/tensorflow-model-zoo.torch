import math
import os
import sys

import h5py
import numpy as np
import tensorflow as tf

sys.path.append('models/research/slim')
from datasets import dataset_utils
from nets.nasnet import nasnet
from nets.nasnet.nasnet import nasnet_large_arg_scope

slim = tf.contrib.slim

url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'


def make_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [math.floor(int(conv_shape[0]) / 2), math.floor(int(conv_shape[1]) / 2)]
    else:
        sys.exit('Invalid padding name ' + padding_name)


def dump_fc(sess, path, name, op_name='BiasAdd'):
    filename = os.path.join(path, name + '.h5')
    if not os.path.exists(filename):
        operation = sess.graph.get_operation_by_name(name + '/' + op_name)

        weight_tensor = sess.graph.get_tensor_by_name(name + '/weights:0')
        weight = weight_tensor.eval()

        biases_tensor = sess.graph.get_tensor_by_name(name + '/biases:0')
        biases = biases_tensor.eval()

        output = operation.outputs[0].eval()
        print('output', output)

        print('save', filename)
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        h5f = h5py.File(filename, 'w')
        # fc
        h5f.create_dataset("weight", data=weight)
        h5f.create_dataset("bias", data=biases)
        h5f.create_dataset("output", data=output)
        h5f.close()


def dump_conv2d(sess, path, name, op_name='Conv2D'):
    filename = os.path.join(path, name + '.h5')
    if not os.path.exists(filename):
        operation = sess.graph.get_operation_by_name(name + '/' + op_name)

        weight_tensor = sess.graph.get_tensor_by_name(name + '/weights:0')
        weight = weight_tensor.eval()

        padding = make_padding(operation.get_attr('padding'), weight_tensor.get_shape())
        stride = operation.get_attr('strides')

        output = operation.outputs[0].eval()

        print('save', filename)
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        h5f = h5py.File(filename, 'w')
        # conv
        h5f.create_dataset("weight", data=weight)
        h5f.create_dataset("stride", data=stride)
        h5f.create_dataset("padding", data=padding)
        h5f.create_dataset("output", data=output)
        h5f.close()


def dump_output(sess, path, name):
    filename = os.path.join(path, name + '_output.h5')
    if not os.path.exists(filename):
        operation = sess.graph.get_operation_by_name(name)
        output = operation.outputs[0].eval()

        print('save', filename)
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset("output", data=output)
        h5f.close()


def dump_separable_conv2d(sess, path, name, op_name='separable_conv2d'):
    filename = os.path.join(path, name + '.h5')
    if not os.path.exists(filename):

        # depthwise
        depthwise_operation = sess.graph.get_operation_by_name(name + '/' + op_name + '/depthwise')

        depthwise_weight_tensor = sess.graph.get_tensor_by_name(name + '/depthwise_weights:0')
        depthwise_weight = depthwise_weight_tensor.eval()

        depthwise_padding = make_padding(depthwise_operation.get_attr('padding'), depthwise_weight_tensor.get_shape())
        depthwise_stride = depthwise_operation.get_attr('strides')

        # pointwise
        pointwise_operation = sess.graph.get_operation_by_name(name + '/' + op_name)

        pointwise_weight_tensor = sess.graph.get_tensor_by_name(name + '/pointwise_weights:0')
        pointwise_weight = pointwise_weight_tensor.eval()

        pointwise_padding = make_padding(pointwise_operation.get_attr('padding'), pointwise_weight_tensor.get_shape())
        pointwise_stride = depthwise_operation.get_attr('strides')

        output = pointwise_operation.outputs[0].eval()

        print('save', filename)
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        h5f = h5py.File(filename, 'w')
        # conv
        h5f.create_dataset("depthwise_weight", data=depthwise_weight)
        h5f.create_dataset("depthwise_stride", data=depthwise_stride)
        h5f.create_dataset("depthwise_padding", data=depthwise_padding)
        h5f.create_dataset("pointwise_weight", data=pointwise_weight)
        h5f.create_dataset("pointwise_stride", data=pointwise_stride)
        h5f.create_dataset("pointwise_padding", data=pointwise_padding)
        h5f.create_dataset("output", data=output)
        h5f.close()


def dump_bn(sess, path, name):
    filename = os.path.join(path, name + '.h5')
    if not os.path.exists(filename):

        gamma = sess.graph.get_tensor_by_name(name + '/gamma:0').eval()
        beta = sess.graph.get_tensor_by_name(name + '/beta:0').eval()
        mean = sess.graph.get_tensor_by_name(name + '/moving_mean:0').eval()
        var = sess.graph.get_tensor_by_name(name + '/moving_variance:0').eval()

        operation = sess.graph.get_operation_by_name(name + '/FusedBatchNorm')
        output = operation.outputs[0].eval()

        print('save', filename)
        parent_dir = os.path.dirname(filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        h5f = h5py.File(filename, 'w')
        # batch norm
        h5f.create_dataset("gamma", data=gamma)
        h5f.create_dataset("beta", data=beta)
        h5f.create_dataset("mean", data=mean)
        h5f.create_dataset("var", data=var)
        # output
        h5f.create_dataset("output", data=output)
        h5f.close()


# def dump_conv2d_0(sess, path, name):
#     filename = os.path.join(path, name + '.h5')
#     if True or not os.path.exists(filename):
#         conv_op_name = name + '/Conv2D'  # remplacer convolution par Conv2D si erreur
#         conv_operation = sess.graph.get_operation_by_name(conv_op_name)
#
#         weights_tensor = sess.graph.get_tensor_by_name(name + '/weights:0')
#         weights = weights_tensor.eval()
#
#         padding = make_padding(conv_operation.get_attr('padding'), weights_tensor.get_shape())
#         strides = conv_operation.get_attr('strides')
#
#         conv_out = sess.graph.get_operation_by_name(conv_op_name).outputs[0].eval()
#
#         gamma = sess.graph.get_tensor_by_name(name + '_bn/gamma:0').eval()
#         beta = sess.graph.get_tensor_by_name(name + '_bn/beta:0').eval()
#         mean = sess.graph.get_tensor_by_name(name + '_bn/moving_mean:0').eval()
#         var = sess.graph.get_tensor_by_name(name + '_bn/moving_variance:0').eval()
#
#         operation = sess.graph.get_operation_by_name(name + '_bn/FusedBatchNorm')
#         output = operation.outputs[0].eval()
#
#         print('save', filename)
#         parent_dir = os.path.dirname(filename)
#         if not os.path.exists(parent_dir):
#             os.makedirs(parent_dir)
#         h5f = h5py.File(filename, 'w')
#         # conv
#         h5f.create_dataset("weights", data=weights)
#         h5f.create_dataset("strides", data=strides)
#         h5f.create_dataset("padding", data=padding)
#         h5f.create_dataset("conv_out", data=conv_out)
#         # batch norm
#         h5f.create_dataset("gamma", data=gamma)
#         h5f.create_dataset("beta", data=beta)
#         h5f.create_dataset("mean", data=mean)
#         h5f.create_dataset("var", data=var)
#         # output
#         h5f.create_dataset("output", data=output)
#         h5f.close()

def dump_comb_iter(sess, path, name, kernel_size_left=None, kernel_size_right=None):
    # left
    if kernel_size_left is not None:
        dump_separable_conv2d(sess=sess, path=path, name=name + '/left/separable_{ks}x{ks}_1'.format(ks=kernel_size_left))
        dump_bn(sess=sess, path=path, name=name + '/left/bn_sep_{ks}x{ks}_1'.format(ks=kernel_size_left))
        dump_separable_conv2d(sess=sess, path=path, name=name + '/left/separable_{ks}x{ks}_2'.format(ks=kernel_size_left))
        dump_bn(sess=sess, path=path, name=name + '/left/bn_sep_{ks}x{ks}_2'.format(ks=kernel_size_left))

        dump_output(sess, path, name + '/left/Relu')
        dump_output(sess, path, name + '/left/Relu_1')

    # right
    if kernel_size_right is not None:
        dump_separable_conv2d(sess=sess, path=path, name=name + '/right/separable_{ks}x{ks}_1'.format(ks=kernel_size_right))
        dump_bn(sess=sess, path=path, name=name + '/right/bn_sep_{ks}x{ks}_1'.format(ks=kernel_size_right))
        dump_separable_conv2d(sess=sess, path=path, name=name + '/right/separable_{ks}x{ks}_2'.format(ks=kernel_size_right))
        dump_bn(sess=sess, path=path, name=name + '/right/bn_sep_{ks}x{ks}_2'.format(ks=kernel_size_right))

        dump_output(sess, path, name + '/right/Relu')
        dump_output(sess, path, name + '/right/Relu_1')

    dump_output(sess, path, name + '/combine/add')


def dump_cell_stem_0(sess, path, name='cell_stem_0'):
    dump_conv2d(sess=sess, path=path, name=name + '/1x1')
    dump_bn(sess=sess, path=path, name=name + '/beginning_bn')

    dump_comb_iter(sess, path, name + '/comb_iter_0', kernel_size_left=5, kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_1', kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_2', kernel_size_right=5)
    dump_comb_iter(sess, path, name + '/comb_iter_3')
    dump_comb_iter(sess, path, name + '/comb_iter_4', kernel_size_left=3)

    dump_output(sess, path, name + '/cell_output/concat')


def dump_cell_stem_1(sess, path, name='cell_stem_1'):
    dump_conv2d(sess=sess, path=path, name=name + '/1x1')
    dump_bn(sess=sess, path=path, name=name + '/beginning_bn')

    dump_conv2d(sess=sess, path=path, name=name + '/path1_conv')
    dump_conv2d(sess=sess, path=path, name=name + '/path2_conv')
    dump_bn(sess=sess, path=path, name=name + '/final_path_bn')

    dump_comb_iter(sess, path, name + '/comb_iter_0', kernel_size_left=5, kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_1', kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_2', kernel_size_right=5)
    dump_comb_iter(sess, path, name + '/comb_iter_3')
    dump_comb_iter(sess, path, name + '/comb_iter_4', kernel_size_left=3)

    dump_output(sess, path, name + '/cell_output/concat')

    dump_output(sess, path, name + '/Relu')
    dump_output(sess, path, name + '/Pad')
    dump_output(sess, path, name + '/strided_slice')
    dump_output(sess, path, name + '/AvgPool')
    dump_output(sess, path, name + '/AvgPool_1')
    dump_output(sess, path, name + '/concat')

    dump_output(sess, path, name + '/Relu_1')


def dump_first_cell(sess, path, name):
    dump_conv2d(sess=sess, path=path, name=name + '/1x1')
    dump_bn(sess=sess, path=path, name=name + '/beginning_bn')

    dump_conv2d(sess=sess, path=path, name=name + '/path1_conv')
    dump_conv2d(sess=sess, path=path, name=name + '/path2_conv')
    dump_bn(sess=sess, path=path, name=name + '/final_path_bn')

    dump_comb_iter(sess, path, name + '/comb_iter_0', kernel_size_left=5, kernel_size_right=3)
    dump_comb_iter(sess, path, name + '/comb_iter_1', kernel_size_left=5, kernel_size_right=3)
    dump_comb_iter(sess, path, name + '/comb_iter_2')
    dump_comb_iter(sess, path, name + '/comb_iter_3')
    dump_comb_iter(sess, path, name + '/comb_iter_4', kernel_size_left=3)

    dump_output(sess, path, name + '/cell_output/concat')

    dump_output(sess, path, name + '/Relu')
    dump_output(sess, path, name + '/Pad')
    dump_output(sess, path, name + '/strided_slice')
    dump_output(sess, path, name + '/AvgPool')
    dump_output(sess, path, name + '/AvgPool_1')
    dump_output(sess, path, name + '/concat')

    dump_output(sess, path, name + '/Relu_1')


def dump_normal_cell(sess, path, name):
    dump_conv2d(sess=sess, path=path, name=name + '/1x1')
    dump_bn(sess=sess, path=path, name=name + '/beginning_bn')

    dump_conv2d(sess=sess, path=path, name=name + '/prev_1x1')
    dump_bn(sess=sess, path=path, name=name + '/prev_bn')

    dump_comb_iter(sess, path, name + '/comb_iter_0', kernel_size_left=5, kernel_size_right=3)
    dump_comb_iter(sess, path, name + '/comb_iter_1', kernel_size_left=5, kernel_size_right=3)
    dump_comb_iter(sess, path, name + '/comb_iter_2')
    dump_comb_iter(sess, path, name + '/comb_iter_3')
    dump_comb_iter(sess, path, name + '/comb_iter_4', kernel_size_left=3)

    dump_output(sess, path, name + '/cell_output/concat')

    dump_output(sess, path, name + '/Relu')
    dump_output(sess, path, name + '/Relu_1')


def dump_reduction_cell(sess, path, name):
    dump_conv2d(sess=sess, path=path, name=name + '/1x1')
    dump_bn(sess=sess, path=path, name=name + '/beginning_bn')

    dump_conv2d(sess=sess, path=path, name=name + '/prev_1x1')
    dump_bn(sess=sess, path=path, name=name + '/prev_bn')

    dump_comb_iter(sess, path, name + '/comb_iter_0', kernel_size_left=5, kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_1', kernel_size_right=7)
    dump_comb_iter(sess, path, name + '/comb_iter_2', kernel_size_right=5)
    dump_comb_iter(sess, path, name + '/comb_iter_3')
    dump_comb_iter(sess, path, name + '/comb_iter_4', kernel_size_left=3)

    dump_output(sess, path, name + '/cell_output/concat')

    dump_output(sess, path, name + '/comb_iter_1/left/MaxPool2D/MaxPool')
    dump_output(sess, path, name + '/comb_iter_2/left/AvgPool2D/AvgPool')
    dump_output(sess, path, name + '/comb_iter_3/right/AvgPool2D/AvgPool')
    dump_output(sess, path, name + '/comb_iter_4/right/MaxPool2D/MaxPool')

    dump_output(sess, path, name + '/Relu')
    dump_output(sess, path, name + '/Relu_1')


def dump_final_layer(sess, path, name):
    dump_output(sess, path, name + '/Relu')
    dump_output(sess, path, name + '/Mean')
    dump_output(sess, path, name + '/predictions')

    dump_fc(sess, path, name+'/FC')



def write_weights(path):
    checkpoints_dir = os.path.join(path, 'checkpoints', 'NASNet-A_Large_331')
    print('checkpoints_dir', checkpoints_dir)
    weights_dir = os.path.join(path, 'weights', 'NASNet-A_Large_331')
    print('weights_dir', weights_dir)

    # download model
    file_checkpoint = os.path.join(checkpoints_dir, 'model.ckpt.index')
    if not tf.gfile.Exists(file_checkpoint):
        tf.gfile.MakeDirs(checkpoints_dir)
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

    file_checkpoint = os.path.join(checkpoints_dir, 'model.ckpt')

    with tf.Graph().as_default():
        # Create model architecture

        image_size = 331
        print('image_size', image_size)
        inputs_np = np.ones((1, image_size, image_size, 3), dtype=np.float32)
        #inputs_np = np.load(weights_dir + '/input.npy')
        print('input', inputs_np.shape)

        inputs = tf.constant(inputs_np, dtype=tf.float32)

        with slim.arg_scope(nasnet_large_arg_scope()):
            logits, _ = nasnet.build_nasnet_large(inputs, num_classes=1001, is_training=False)

        with tf.Session() as sess:
            # Initialize model
            init_fn = slim.assign_from_checkpoint_fn(file_checkpoint, slim.get_model_variables())
            init_fn(sess)

            # Display model variables
            for v in slim.get_model_variables():
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))

            # Create graph
            os.system("rm -rf logs")
            os.system("mkdir -p logs")

            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

            # conv0
            dump_conv2d(sess=sess, path=weights_dir, name='conv0')
            dump_bn(sess=sess, path=weights_dir, name='conv0_bn')

            # cell_stem
            dump_cell_stem_0(sess=sess, path=weights_dir, name='cell_stem_0')
            dump_cell_stem_1(sess=sess, path=weights_dir, name='cell_stem_1')

            dump_first_cell(sess=sess, path=weights_dir, name='cell_0')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_1')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_2')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_3')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_4')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_5')

            dump_reduction_cell(sess=sess, path=weights_dir, name='reduction_cell_0')

            dump_first_cell(sess=sess, path=weights_dir, name='cell_6')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_7')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_8')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_9')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_10')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_11')

            dump_reduction_cell(sess=sess, path=weights_dir, name='reduction_cell_1')

            dump_first_cell(sess=sess, path=weights_dir, name='cell_12')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_13')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_14')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_15')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_16')
            dump_normal_cell(sess=sess, path=weights_dir, name='cell_17')

            dump_final_layer(sess, weights_dir, name='final_layer')


def main():
    tmpdir = '/tmp/tf-models'
    #tmpdir = '/Users/thibaut/Documents/lip6/project/tmp/models'
    write_weights(tmpdir)


if __name__ == '__main__':
    main()
