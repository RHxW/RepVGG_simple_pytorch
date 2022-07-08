from collections import OrderedDict

import torch

from model import *


class RepVGGBaseMTrain(nn.Module):
    def __init__(self, block_num, channel_num):
        """
        this is a modified version model for train  # TODO add groupwise conv
        :param block_num:
        :param channel_num:
        """
        super(RepVGGBaseMTrain, self).__init__()
        assert len(channel_num) - len(block_num) == 1
        self.channel_num = channel_num
        self.block_num = block_num

        layers = []
        for i in range(len(block_num)):
            layers.append(("stage_%d" % (i + 1),
                           make_layer(self.block_num[i], self.channel_num[i], self.channel_num[i + 1], train=True)))

        self.net = nn.Sequential(OrderedDict(layers))
        # self.stage_1 = make_layer(self.block_num[0], self.channel_num[0], self.channel_num[1], train=True)
        # self.stage_2 = make_layer(self.block_num[1], self.channel_num[1], self.channel_num[2], train=True)
        # self.stage_3 = make_layer(self.block_num[2], self.channel_num[2], self.channel_num[3], train=True)
        # self.stage_4 = make_layer(self.block_num[3], self.channel_num[3], self.channel_num[4], train=True)
        # self.stage_5 = make_layer(self.block_num[4], self.channel_num[4], self.channel_num[5], train=True)

    def forward(self, x):
        out = self.net(x)
        return out


class RepVGGBaseMInfer(nn.Module):
    def __init__(self, block_num, channel_num):
        """

        :param block_num:
        :param channel_num:
        """
        super(RepVGGBaseMInfer, self).__init__()
        assert len(channel_num) - len(block_num) == 1
        self.channel_num = channel_num
        self.block_num = block_num

        layers = []
        for i in range(len(block_num)):
            layers.append(("stage_%d" % (i + 1),
                           make_layer(self.block_num[i], self.channel_num[i], self.channel_num[i + 1], train=False)))

        self.net = nn.Sequential(OrderedDict(layers))

    def load_from_train_model(self, train_model):
        assert isinstance(train_model, RepVGGBaseMTrain)
        torch.set_grad_enabled(False)
        ks_train = list(train_model.net._modules.keys())
        layers = []
        for k in ks_train:
            sub_model = train_model.net._modules[k]
            layers.append((k, self._load_stage_from_train_model(sub_model)))

        # self.stage_1 = self._load_stage_from_train_model(train_model.stage_1)
        # self.stage_2 = self._load_stage_from_train_model(train_model.stage_2)
        # self.stage_3 = self._load_stage_from_train_model(train_model.stage_3)
        # self.stage_4 = self._load_stage_from_train_model(train_model.stage_4)
        # self.stage_5 = self._load_stage_from_train_model(train_model.stage_5)
        self.net = nn.Sequential(OrderedDict(layers))

    def _load_stage_from_train_model(self, stage_x):
        blocks = []
        ks = list(stage_x._modules.keys())
        for k in ks:
            train_block = stage_x._modules[k]
            _block = RepVGGBlockInfer(train_block.in_c, train_block.out_c, train_block.downsample)
            _block.load_weight_from_train_block(train_block)
            blocks.append((k, _block))
        return nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        out = self.net(x)
        return out


def make_layer(layer_block_num, in_c, out_c, train=True):
    """

    :param layer_block_num:
    :param in_c:
    :param out_c:
    :return:
    """
    if train:
        _block = RepVGGBlockTrain
    else:
        _block = RepVGGBlockInfer
    blocks = []
    for i in range(layer_block_num):
        downsample = False
        c1 = c2 = out_c
        if i == 0:
            downsample = True
            c1 = in_c
        blocks.append((str(i), _block(c1, c2, downsample)))

    return nn.Sequential(OrderedDict(blocks))


if __name__ == "__main__":
    block_num = [1, 2, 8]
    channel_num = [3, 64, 128, 256]

    model = RepVGGBaseMTrain(block_num, channel_num)
    model_i = RepVGGBaseMInfer(block_num, channel_num)
    # torch.save(model.state_dict(),'1.pth')
    # torch.save(model_i.state_dict(),'2.pth')
    model_i.load_from_train_model(model)
    model_i.eval()
    stage1 = model.net.stage_1
    stage11 = model_i.net.stage_1
    stage1.eval()
    stage11.eval()
    # torch.save(model_i.state_dict(),'3.pth')
    a = torch.ones([1, 3, 16, 16])
    b = model(a)
    print(b.shape)
