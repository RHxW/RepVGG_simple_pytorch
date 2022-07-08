from collections import OrderedDict

import torch

from model import *


class RepVGGBaseATrain(nn.Module):
    def __init__(self, a=0.75, b=2.5):
        """
        this is the model A for train  # TODO add groupwise conv
        :param a: We use multiplier `a` to scale the first four stages and `b` for the last stage,
        and usually set `b > a` because we desire the last layer to have richer features for the
        classification or other down-stream tasks
        :param b:
        """
        super(RepVGGBaseATrain, self).__init__()

        # self.channel_base = [64, 128, 256, 512]
        self.a = a
        self.b = b
        self.channel_num = [3, min(64, int(64 * a)), int(64 * a), int(128 * a), int(256 * a), int(512 * b)]
        self.block_num = [1, 2, 4, 14, 1]

        self.stage_1 = make_layer(self.block_num[0], self.channel_num[0], self.channel_num[1], train=True)
        self.stage_2 = make_layer(self.block_num[1], self.channel_num[1], self.channel_num[2], train=True)
        self.stage_3 = make_layer(self.block_num[2], self.channel_num[2], self.channel_num[3], train=True)
        self.stage_4 = make_layer(self.block_num[3], self.channel_num[3], self.channel_num[4], train=True)
        self.stage_5 = make_layer(self.block_num[4], self.channel_num[4], self.channel_num[5], train=True)

    def forward(self, x):
        out = self.stage_1(x)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        out = self.stage_5(out)
        return out


class RepVGGBaseAInfer(nn.Module):
    def __init__(self, a=0.75, b=2.5):
        """

        :param a: We use multiplier `a` to scale the first four stages and `b` for the last stage,
        and usually set `b > a` because we desire the last layer to have richer features for the
        classification or other down-stream tasks
        :param b:
        """
        super(RepVGGBaseAInfer, self).__init__()
        self.a = a
        self.b = b
        self.channel_num = [3, min(64, int(64 * a)), int(64 * a), int(128 * a), int(256 * a), int(512 * b)]
        self.block_num = [1, 2, 4, 14, 1]

        self.stage_1 = make_layer(self.block_num[0], self.channel_num[0], self.channel_num[1], train=False)
        self.stage_2 = make_layer(self.block_num[1], self.channel_num[1], self.channel_num[2], train=False)
        self.stage_3 = make_layer(self.block_num[2], self.channel_num[2], self.channel_num[3], train=False)
        self.stage_4 = make_layer(self.block_num[3], self.channel_num[3], self.channel_num[4], train=False)
        self.stage_5 = make_layer(self.block_num[4], self.channel_num[4], self.channel_num[5], train=False)

    def load_from_train_model(self, train_model):
        assert isinstance(train_model, RepVGGBaseATrain)
        torch.set_grad_enabled(False)
        self.stage_1 = self._load_stage_from_train_model(train_model.stage_1)
        self.stage_2 = self._load_stage_from_train_model(train_model.stage_2)
        self.stage_3 = self._load_stage_from_train_model(train_model.stage_3)
        self.stage_4 = self._load_stage_from_train_model(train_model.stage_4)
        self.stage_5 = self._load_stage_from_train_model(train_model.stage_5)

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
        out = self.stage_1(x)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        out = self.stage_5(out)
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
    model = RepVGGBaseATrain(0.75, 1)
    model_i = RepVGGBaseAInfer(0.75, 2)
    # torch.save(model.state_dict(),'1.pth')
    # torch.save(model_i.state_dict(),'2.pth')
    # model_i.load_from_train_model(model)
    # torch.save(model_i.state_dict(),'3.pth')
    a = torch.ones([1, 3, 368, 368])
    b = model(a)
    print(b.shape)
