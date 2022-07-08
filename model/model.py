import torch
from torch import nn
from itertools import product


class RepVGGBlockTrain(nn.Module):
    """
    this is the block for train stage
    """

    def __init__(self, in_c, out_c, downsample=False):
        super(RepVGGBlockTrain, self).__init__()
        self.downsample = downsample
        if downsample:
            stride = 2
        else:
            stride = 1
        # if stride not in [1, 2]:
        #     raise RuntimeError("stride should be in [1, 2](for the first block of each stage, stride=2, else stride=1)")
        self.idt = False
        self.in_c = in_c
        self.out_c = out_c
        if in_c == out_c and not downsample:
            self.idt = True
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn_idt = None
        if self.idt:
            self.bn_idt = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(self.conv(x)) + self.bn1(self.conv1(x))
        if self.idt:
            out += self.bn_idt(x)
        out = self.relu(out)
        return out


class RepVGGBlockInfer(nn.Module):
    """
    this is the block for test/inference stage
    """

    def __init__(self, in_c, out_c, downsample=False):
        super(RepVGGBlockInfer, self).__init__()
        self.downsample = downsample
        if downsample:
            stride = 2
        else:
            stride = 1
        # if stride not in [1, 2]:
        #     raise RuntimeError("stride should be in [1, 2](for the first block of each stage, stride=2, else stride=1)")
        self.in_c = in_c
        self.out_c = out_c
        # self.idt = False
        # if in_c == out_c:
        #     self.idt = True
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def load_weight_from_train_block(self, train_block):
        assert isinstance(train_block, RepVGGBlockTrain)
        if not conv_align(self.conv, train_block.conv):
            raise RuntimeError("the conv layer of two blocks does not align!")
        self.conv = block_reparameterize(train_block)

    def forward(self, x):
        return self.relu(self.conv(x))


def conv_align(conv1: nn.Conv2d, conv2: nn.Conv2d):
    """
    see if two conv layers' parameters match with each other
    :param conv1:
    :param conv2:
    :return:
    """
    flg = 1
    flg *= conv1.in_channels == conv2.in_channels
    flg *= conv1.out_channels == conv2.out_channels
    flg *= conv1.kernel_size == conv2.kernel_size
    flg *= conv1.stride == conv2.stride
    flg *= conv1.padding == conv2.padding
    return flg


def block_reparameterize(train_block):
    """
    to re-parameterize a block from train to inference
    :param train_block:
    :return:
    """
    assert isinstance(train_block, RepVGGBlockTrain)
    in_c = train_block.conv.in_channels
    out_c = train_block.conv.out_channels
    new_conv = nn.Conv2d(in_c, out_c, train_block.conv.kernel_size, stride=train_block.conv.stride,
                         padding=train_block.conv.padding)
    new_conv.weight.data *= 0
    new_conv.bias.data *= 0

    # 1. convert conv layers
    conv = train_block.conv
    conv1 = conv_one_to_three(train_block.conv1)
    # 2. fuse bn layers
    conv_new = conv_bn_fuse(conv, train_block.bn)
    conv1_new = conv_bn_fuse(conv1, train_block.bn1)
    # 3. add
    new_conv.weight.data += conv_new.weight.data
    new_conv.bias.data += conv_new.bias.data
    new_conv.weight.data += conv1_new.weight.data
    new_conv.bias.data += conv1_new.bias.data
    if train_block.idt:
        idt_conv = idt_conv_three(in_c, out_c)
        idt_conv = conv_bn_fuse(idt_conv, train_block.bn_idt)
        new_conv.weight.data += idt_conv.weight.data
        new_conv.bias.data += idt_conv.bias.data

    return new_conv


def conv_bn_fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    fuse batchnorm with conv2d
    output new weight and bias for a new conv2d layer
    :param conv:
    :param bn:
    :return:
    """
    mu = bn.running_mean
    sigma = bn.running_var ** 0.5
    gamma = bn.weight
    beta = bn.bias
    k = gamma / sigma

    w = conv.weight
    b = conv.bias

    new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                         padding=conv.padding)

    assert w.shape[0] == k.shape[0]
    for i in range(w.shape[0]):
        new_conv.weight.data[i] = k[i] * w[i]
        new_conv.bias.data[i] = k[i] * (b[i] - mu[i]) + beta[i]

    return new_conv


def conv_one_to_three(conv: nn.Conv2d):
    """
    convert 1*1 conv to 3*3 conv
    :param conv:
    :return:
    """
    w = conv.weight
    b = conv.bias
    out_c, in_c, k, k = w.shape
    assert k == 1
    weight3 = torch.zeros([out_c, in_c, 3, 3]).to(torch.float32)

    for i, j in product(range(out_c), range(in_c)):
        weight3[i, j, 1, 1] = w[i, j, 0, 0]
    new_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3)
    new_conv.weight.data = weight3
    new_conv.bias.data = b
    return new_conv


def idt_conv_three(in_c, out_c):
    """
    convert identity branch to a 3*3 conv layer
    :param in_c:
    :param out_c:
    :return:
    """
    assert in_c == out_c
    conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3)
    weight = torch.zeros([out_c, in_c, 3, 3])
    for i in range(in_c):
        weight[i, i, 1, 1] = 1.
    conv.weight.data = weight
    return conv


if __name__ == "__main__":
    block = RepVGGBlockTrain(3, 6, downsample=False)
    x = torch.ones([1, 3, 10, 10])
    y = block(x)
    print(y.shape)
