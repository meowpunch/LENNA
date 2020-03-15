# NAS Recasting source code
# shorm21

from Recasting_ver.models.normal_nets.proxyless_nets import *


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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.shortcut is not None :
            res = self.shortcut(x)
        else :
            res = x

        x = self.conv1(x)
        x = self.conv2(x)

        return self.relu(x + res)

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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.shortcut is not None :
            res = self.shortcut(x)
        else :
            res = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.relu(x + res)

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

# Modifier : shorm21
class BaseRecastingNet(MyNetwork):

    def __init__(self, first_conv, blocks, classifier):
        super(BaseRecastingNet, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
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

def build_resblock(input_channel, output_channel, expansion=4, block_type='basic'):
    if block_type == 'basic' :
        if input_channel == output_channel :
            stride = 1
            shortcut = None
        else :
            stride = 2
            shortcut = ConvLayer(input_channel, output_channel,
                                 kernel_size=1, stride=2, use_bn=True, act_func=None, ops_order='weight_bn_act')
        conv1 = ConvLayer(input_channel, output_channel,
                          kernel_size=3, stride=stride, use_bn=True, act_func='relu', ops_order='weight_bn_act') 
        conv2 = ConvLayer(output_channel, output_channel,
                          kernel_size=3, stride=1, use_bn=True, act_func=None, ops_order='weight_bn_act') 

        return ResidualBlock(conv1, conv2, shortcut)
    elif block_type == 'bottleneck' :
        if input_channel == output_channel :
            stride = 1
            shortcut = None
        else :
            stride = 2
            shortcut = ConvLayer(input_channel, output_channel,
                                 kernel_size=1, stride=2, use_bn=True, act_func=None, ops_order='weight_bn_act')
        conv1 = ConvLayer(input_channel, output_channel*expansion,
                          kernel_size=1, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act') 
        conv2 = ConvLayer(output_channel*expansion, output_channel*expansion,
                          kernel_size=3, stride=stride, use_bn=True, act_func='relu', ops_order='weight_bn_act') 
        conv3 = ConvLayer(output_channel*expansion, output_channel,
                          kernel_size=1, stride=1, use_bn=True, act_func=None, ops_order='weight_bn_act') 

        return BottleneckBlock(conv1, conv2, conv3, shortcut)
    else :
        raise NotImplementedError

def build_resnet(num_blocks = [9, 9, 9], num_classes = 10, block_type = 'basic'):
    input_channel = make_divisible(16, 8)
    first_conv = ConvLayer(
        3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
    )
    blocks = []
    output_channel = input_channel
    for i, nb in enumerate(num_blocks):
        input_channel = output_channel
        
        if i == 0 :  # First block (keep dimension)
            b = build_resblock(input_channel, output_channel, block_type=block_type)
            blocks += [b]
        else :       # Reduction blocks
            output_channel = input_channel * 2
            b = build_resblock(input_channel, output_channel, block_type=block_type)
            blocks += [b]

        for _ in range(nb-1): # Normal blocks
            b = build_resblock(output_channel, output_channel, block_type=block_type)
            blocks += [b]
    classifier = LinearLayer(output_channel, num_classes)

    return DartsRecastingNet(first_conv, blocks, classifier)

def build_densenet():
    raise NotImplementedError

def build_mbv2block(input_channel, output_channel, expand_ratio=6):
    if input_channel == output_channel :
        stride = 1
        shortcut = None
    else :
        stride = 2
        shortcut = ConvLayer(input_channel, output_channel,
                             kernel_size=1, stride=2, use_bn=True, act_func=None, ops_order='weight_bn_act')
    mobile_inverted_conv = MBInvertedConvLayer(input_channel, output_channel, stride=stride, expand_ratio=expand_ratio)

    return MobileInvetedResidualBlock(mobile_inverted_conv, shortcut)


def build_mobilenetv2():
    input_channel = make_divisible(16, 8)
    first_conv = ConvLayer(
        3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
    )
    blocks = []
    output_channel = input_channel
    for i, nb in enumerate(num_blocks):
        input_channel = output_channel
        
        if i == 0 :  # First block (keep dimension)
            b = build_mbv2block(input_channel, output_channel, block_type=block_type)
            blocks += [b]
        else :       # Reduction blocks
            output_channel = input_channel * 2
            b = build_mbv2block(input_channel, output_channel, block_type=block_type)
            blocks += [b]

        for _ in range(nb-1): # Normal blocks
            b = build_mbv2block(output_channel, output_channel, block_type=block_type)
            blocks += [b]
    classifier = LinearLayer(output_channel, n_classes)

    return DartsRecastingNet(first_conv, blocks, classifier)

