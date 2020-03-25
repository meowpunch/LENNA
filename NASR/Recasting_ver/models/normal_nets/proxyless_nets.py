# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import torchprof

from Recasting_ver.modules.layers import *
import json

from util.latency import get_time
from util.logger import init_logger


def proxyless_base(net_config=None, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0):
    assert net_config is not None, 'Please input a network config'
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net

# Modifier : shorm21
# def rescasting_base(net_config=None, n_classes=10, bn_param=(0.1, 1e-3), dropout_rate=0):
#     assert net_config is not None, 'Please input a network config'
#     net_config_path = download_url(net_config)
#     net_config_json = json.load(open(net_config_path, 'r'))
# 
#     net_config_json['classifier']['out_features'] = n_classes
#     net_config_json['classifier']['dropout_rate'] = dropout_rate
# 
#     net = ProxylessNASNets.build_from_config(net_config_json)
#     net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
# 
#     def build_from_config(config):
#         first_conv = set_layer_from_config(config['first_conv'])
#         classifier = set_layer_from_config(config['classifier'])
#         blocks = []
#         for block_config in config['blocks']:
#             blocks.append(DartsRecastingBlock.build_from_config(block_config))
# 
#         net = DartsRecastingNet(first_conv, blocks, classifier)
#         if 'bn' in config:
#             net.set_bn_param(**config['bn'])
#         else:
#             net.set_bn_param(momentum=0.1, eps=1e-3)
# 
#         return net
# 
#     return net

class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

class ResidualBlock(MyModule):

    def __init__(self, conv1, conv2, shortcut):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is None :
            res = self.shortcut(x)
        else :
            res = x

        x = conv1(x)
        x = conv2(x)

        return x + res 

    @property
    def module_str(self):
        return '(%s, %s, %s)' % (
            self.conv1.module_str, self.conv2.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': ResidualBlock.__name__,
            'conv1': self.conv1.config,
            'conv2': self.conv2.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv1 = set_layer_from_config(config['conv1'])
        conv2 = set_layer_from_config(config['conv2'])
        shortcut = set_layer_from_config(config['shortcut'])
        return ResidualBlock(conv1, conv2, shortcut)

    def get_flops(self, x):
        flops1, conv1_x = self.conv1.get_flops(x)
        flops2, _ = self.conv2.get_flops(conv1_x)

        if self.shortcut:
            flops3, _ = self.shortcut.get_flops(x)
        else:
            flops3 = 0

        return flops1 + flops2 + flops3, self.forward(x)

class BottleneckBlock(MyModule):

    def __init__(self, conv1, conv2, conv3, shortcut):
        super(BottleneckBlock, self).__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is None :
            res = self.shortcut(x)
        else :
            res = x

        x = conv1(x)
        x = conv2(x)
        x = conv3(x)

        return x + res 

    @property
    def module_str(self):
        return '(%s, %s, %s)' % (
            self.conv1.module_str, self.conv2.module_str, self.conv3.module_str,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': BottleneckBlock.__name__,
            'conv1': self.conv1.config,
            'conv2': self.conv2.config,
            'conv3': self.conv3.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv1 = set_layer_from_config(config['conv1'])
        conv2 = set_layer_from_config(config['conv2'])
        conv3 = set_layer_from_config(config['conv3'])
        shortcut = set_layer_from_config(config['shortcut'])
        return BottleneckBlock(conv1, conv2, conv3, shortcut)

    def get_flops(self, x):
        flops1, conv1_x = self.conv1.get_flops(x)
        flops2, conv2_x = self.conv2.get_flops(conv1_x)
        flops3, _ = self.conv3.get_flops(conv2_x)

        if self.shortcut:
            flops4, _ = self.shortcut.get_flops(x)
        else:
            flops4 = 0

        return flops1 + flops2 + flops3 + flops4, self.forward(x)


## Modifier : shorm21
class DartsRecastingBlock(MyModule):

    def __init__(self, layer_list):
        super(DartsRecastingBlock, self).__init__()

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, x):
        x_list = [x]

        for op_list in self.layer_list :
            x_out = op_list[0](x_list[0])
            for x_in, op in zip(x_list[1:], op_list[1:]):
                x_out = x_out + op(x_in)
            x_list += [x_out]
        return x_list[-1]

    @property
    def module_str(self):
        str_ = '['
        for op_list in self.layer_list :
            sub_str = '['
            for op in op_list:
                sub_str += op.module_str + ', '
            sub_str += ']'
            str_ += sub_str + ', '
        str_ += ']'

        return 'Recasting (%s)' % (
                str_
        )
    
    @property
    def arch_params(self):
        str_ = '['
        for op_list in self.layer_list :
            sub_str = '['
            for op in op_list:
                sub_str += str(op.arch_params) + ', '
            sub_str += ']'
            str_ += sub_str + ', '
        str_ += ']'

        return 'Recasting (%s)' % (
                str_
        )

    @property
    def config(self):
        config_list = []
        for op_list in self.layer_list :
            configs = []
            for op in op_list:
                configs += [op.config]
            config_list += [configs]

        return {
            'name': DartsRecastingBlock.__name__,
            'layer_list': config_list,
        }

    @staticmethod
    def build_from_config(config):
        layer_list = []
        for op_list in config['layer_list']:
            layer = []
            for op in op_list:
                l = set_layer_from_config(op)
                layer += [l]
            layer_list += [nn.ModuleList(layer)]

        return DartsRecastingBlock(layer_list)

    def get_flops(self, x):
        flops = 0
        x_list = [x]
        for op_list in self.layer_list :
            f, x_out = op_list[0].get_flops(x_list[0])
            flops += f
            for x_in, op in zip(x_list[1:], op_list[1:]) :
                f_t, x_out_t = op.get_flops(x_in)
                f += f_t
                x_out += x_out_t
                flops += f

            x_list += [x_out]

        return flops, self.forward(x)

class ProxylessNASNets(MyNetwork):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.feature_mix_layer.get_flops(x)
        flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x

# Modifier : shorm21
class DartsRecastingNet(MyNetwork):
    def __init__(self, first_conv, blocks, classifier):
        super(DartsRecastingNet, self).__init__()

        self.latency = None
        self.logger = init_logger()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)

        for block in self.blocks:
            # with torchprof.Profile(block, use_cuda=True) as prof:
            x = block(x)
            # self.latency = sum(get_time(prof))
            # self.logger.debug(sum(get_time(prof)))
            # self.logger.debug(prof.display(show_events=False))

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def forward_recasting(self, x, block_idx = None):
        x = self.first_conv(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == block_idx :
                return x
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def forward_remain(self, x, block_idx = None):
        for i, block in enumerate(self.blocks):
            if i > block_idx :
                x = block(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': DartsRecastingNet.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(DartsRecastingBlock.build_from_config(block_config))

        net = DartsRecastingNet(first_conv, blocks, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop, x = self.first_conv.get_flops(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.get_flops(x)
        flop += delta_flop
        return flop, x
