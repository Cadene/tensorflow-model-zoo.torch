import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#from models.nasnet.debug import read_output


######################################################################
## Load parameters from HDF5 to Dict
######################################################################
#from models.util import save_model


def load_conv2d(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weight'][()]).permute(3, 2, 0, 1)
    try:
        state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['bias'][()])
    except:
        pass
    h5f.close()


def load_linear(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weight'][()]).t()
    try:
        state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['bias'][()])
    except:
        pass
    h5f.close()


def load_bn(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['gamma'][()])
    state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['beta'][()])
    state_dict[name_pth + '.running_mean'] = torch.from_numpy(h5f['mean'][()])
    state_dict[name_pth + '.running_var'] = torch.from_numpy(h5f['var'][()])
    h5f.close()


def load_separable_conv2d(state_dict, path, name_pth, name_tf):
    h5f = h5py.File(path + '/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.depthwise_conv2d.weight'] = torch.from_numpy(h5f['depthwise_weight'][()]).permute(2, 3, 0,
                                                                                                              1)
    try:
        state_dict[name_pth + '.depthwise_conv2d.bias'] = torch.from_numpy(h5f['depthwise_bias'][()])
    except:
        pass
    state_dict[name_pth + '.pointwise_conv2d.weight'] = torch.from_numpy(h5f['pointwise_weight'][()]).permute(3, 2, 0,
                                                                                                              1)
    try:
        state_dict[name_pth + '.pointwise_conv2d.bias'] = torch.from_numpy(h5f['pointwise_bias'][()])
    except:
        pass
    h5f.close()


def load_cell_branch(state_dict, path, name_pth, name_tf, branch, kernel_size):
    load_separable_conv2d(state_dict, path, name_pth=name_pth + '_{branch}.separable_1'.format(branch=branch),
                          name_tf=name_tf + '/{branch}/separable_{ks}x{ks}_1'.format(branch=branch, ks=kernel_size))
    load_bn(state_dict, path, name_pth=name_pth + '_{branch}.bn_sep_1'.format(branch=branch),
            name_tf=name_tf + '/{branch}/bn_sep_{ks}x{ks}_1'.format(branch=branch, ks=kernel_size))
    load_separable_conv2d(state_dict, path, name_pth=name_pth + '_{branch}.separable_2'.format(branch=branch),
                          name_tf=name_tf + '/{branch}/separable_{ks}x{ks}_2'.format(branch=branch, ks=kernel_size))
    load_bn(state_dict, path, name_pth=name_pth + '_{branch}.bn_sep_2'.format(branch=branch),
            name_tf=name_tf + '/{branch}/bn_sep_{ks}x{ks}_2'.format(branch=branch, ks=kernel_size))


def load_cell_stem_0(state_dict, path, name_pth='cell_stem_0', name_tf='cell_stem_0'):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_cell_stem_1(state_dict, path, name_pth='cell_stem_1', name_tf='cell_stem_1'):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    load_conv2d(state_dict, path, name_pth=name_pth + '.path_1.conv', name_tf=name_tf + '/path1_conv')
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_2.conv', name_tf=name_tf + '/path2_conv')
    load_bn(state_dict, path, name_pth=name_pth + '.final_path_bn', name_tf=name_tf + '/final_path_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_first_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # other path
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_1.conv', name_tf=name_tf + '/path1_conv')
    load_conv2d(state_dict, path, name_pth=name_pth + '.path_2.conv', name_tf=name_tf + '/path2_conv')
    load_bn(state_dict, path, name_pth=name_pth + '.final_path_bn', name_tf=name_tf + '/final_path_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=3)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=3)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_normal_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # conv prev_1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.conv', name_tf=name_tf + '/prev_1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.bn', name_tf=name_tf + '/prev_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=3)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=3)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load_reduction_cell(state_dict, path, name_pth, name_tf):
    # conv 1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_1x1.conv', name_tf=name_tf + '/1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_1x1.bn', name_tf=name_tf + '/beginning_bn')

    # conv prev_1x1
    load_conv2d(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.conv', name_tf=name_tf + '/prev_1x1')
    load_bn(state_dict, path, name_pth=name_pth + '.conv_prev_1x1.bn', name_tf=name_tf + '/prev_bn')

    # comb_iter_0
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='left', kernel_size=5)
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_0', name_tf=name_tf + '/comb_iter_0',
                     branch='right', kernel_size=7)

    # comb_iter_1
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_1', name_tf=name_tf + '/comb_iter_1',
                     branch='right', kernel_size=7)

    # comb_iter_2
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_2', name_tf=name_tf + '/comb_iter_2',
                     branch='right', kernel_size=5)

    # comb_iter_4
    load_cell_branch(state_dict, path, name_pth=name_pth + '.comb_iter_4', name_tf=name_tf + '/comb_iter_4',
                     branch='left', kernel_size=3)


def load(path):
    state_dict = {}

    # block1
    load_conv2d(state_dict, path, name_pth='conv0.conv', name_tf='conv0')
    load_bn(state_dict, path, name_pth='conv0.bn', name_tf='conv0_bn')

    # cell_stem
    load_cell_stem_0(state_dict, path, 'cell_stem_0', 'cell_stem_0')
    load_cell_stem_1(state_dict, path, 'cell_stem_1', 'cell_stem_1')

    load_first_cell(state_dict, path, 'cell_0', 'cell_0')
    for i in range(1, 6):
        load_normal_cell(state_dict, path, 'cell_' + str(i), 'cell_' + str(i))

    load_reduction_cell(state_dict, path, 'reduction_cell_0', 'reduction_cell_0')

    load_first_cell(state_dict, path, 'cell_6', 'cell_6')
    for i in range(7, 12):
        load_normal_cell(state_dict, path, 'cell_' + str(i), 'cell_' + str(i))

    load_reduction_cell(state_dict, path, 'reduction_cell_1', 'reduction_cell_1')

    load_first_cell(state_dict, path, 'cell_12', 'cell_12')
    for i in range(13, 18):
        load_normal_cell(state_dict, path, 'cell_' + str(i), 'cell_' + str(i))

    load_linear(state_dict, path, 'linear', 'final_layer/FC')

    return state_dict


class MaxPoolPad(nn.Module):
    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class AvgPoolPad(nn.Module):
    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:]
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):
    def __init__(self):
        super(CellStem0, self).__init__()

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(42, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(42, 42, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(96, 42, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(42, 42, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):
    def __init__(self):
        super(CellStem1, self).__init__()

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(168, 84, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(84, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(96, 42, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(84, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(84, 84, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(84, 84, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()

        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()

        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):
    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()

        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetALarge(nn.Module):
    def __init__(self, num_classes=1001, aux_logits=True, transform_input=True):
        super(NASNetALarge, self).__init__()

        self.num_classes = num_classes
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=0, stride=2,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True))

        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()

        self.cell_0 = FirstCell(in_channels_left=168, out_channels_left=84,
                                in_channels_right=336, out_channels_right=168)
        self.cell_1 = NormalCell(in_channels_left=336, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_2 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_3 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_4 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)
        self.cell_5 = NormalCell(in_channels_left=1008, out_channels_left=168,
                                 in_channels_right=1008, out_channels_right=168)

        self.reduction_cell_0 = ReductionCell0(in_channels_left=1008, out_channels_left=336,
                                               in_channels_right=1008, out_channels_right=336)

        self.cell_6 = FirstCell(in_channels_left=1008, out_channels_left=168,
                                in_channels_right=1344, out_channels_right=336)
        self.cell_7 = NormalCell(in_channels_left=1344, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_8 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_9 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2016, out_channels_right=336)
        self.cell_10 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                  in_channels_right=2016, out_channels_right=336)
        self.cell_11 = NormalCell(in_channels_left=2016, out_channels_left=336,
                                  in_channels_right=2016, out_channels_right=336)

        self.reduction_cell_1 = ReductionCell1(in_channels_left=2016, out_channels_left=672,
                                               in_channels_right=2016, out_channels_right=672)

        self.cell_12 = FirstCell(in_channels_left=2016, out_channels_left=336,
                                 in_channels_right=2688, outt_channels_right=672)
        self.cell_13 = NormalCell(in_channels_left=2688, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_14 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_15 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_16 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)
        self.cell_17 = NormalCell(in_channels_left=4032, out_channels_left=672,
                                  in_channels_right=4032, out_channels_right=672)

        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(4032, self.num_classes)

    def features(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)

        return x_cell_17

    def classifier(self, x):
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_and_save_model(path):
    path_weights = os.path.join(path, 'weights', 'NASNet-A_Large_331')
    model = NASNetALarge()
    state_dict = load(path_weights)
    model.load_state_dict(state_dict)
    filename_model = os.path.join(path, 'pytorch', 'nasnet_a_large.pth')
    os.system('mkdir -p '+path+'/pytorch')
    torch.save(model.state_dict(), filename_model)
    return model


def main():
    #path = '/local/durandt/tmp/models'
    path = '/tmp/tf-models'
    #path = '/Users/thibaut/Documents/lip6/project/tmp/models'
    model = build_and_save_model(path)
    model.transform_input = False
    model.eval()

    print(model)

    input = torch.autograd.Variable(torch.ones(1, 3, 331, 331))
    output = model.forward(input)
    print('output', output)




if __name__ == '__main__':
    main()
