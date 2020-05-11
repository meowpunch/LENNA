from Recasting_ver.models.normal_nets.proxyless_nets import DartsRecastingNet, DartsRecastingBlock
from Recasting_ver.modules.layers import nn, ConvLayer, LinearLayer
from Recasting_ver.modules.mix_op import MixedEdge, build_candidate_ops


class LennaNet(DartsRecastingNet):
    """
        SuperDartsRecastingNet:
            first_conv -> Blocks(Cells) -> pool -> classifier

        LennaNet:
            first conv -> one block -> pool -> classifier
    """

    def __init__(self, num_layers,
                 normal_ops, reduction_ops, block_type,
                 input_channel, n_classes=1000,
                 bn_param=(0.1, 1e-3), dropout_rate=0,
                 ):
        self._redundant_modules = None

        """
            first_conv.input_channel <- 3 (RGB)
            first_conv.output_channel <- b_input_channel 
        """
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
        )

        output_channel = input_channel
        if block_type:
            edges = self.build_normal_layers(normal_ops, input_channel, output_channel, num_layers)
            block = [DartsRecastingBlock(edges)]
        else:
            output_channel = input_channel * 2
            edges = self.build_reduction_layers(reduction_ops, normal_ops, input_channel, output_channel,
                                                num_layers)
            block = [DartsRecastingBlock(edges)]

        classifier = LinearLayer(output_channel, n_classes, dropout_rate=dropout_rate)
        super(LennaNet, self).__init__(first_conv, block, classifier)

    def init_arch_params(self, init_type='uniform', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    @staticmethod
    def build_normal_layers(candidate_ops, input_channel, out_channel, num_layers):
        """
            TODO: apply MixedEdge_v2
            build normal layers for one block.
        :return: layer_list will be args for DartsRecastingBlock(arg)
        """
        layer_list = []
        for num_edges in range(num_layers):
            layer = []
            for _ in range(num_edges + 1):
                edge = MixedEdge(candidate_ops=build_candidate_ops(candidate_ops,
                                                                   input_channel,
                                                                   out_channel,
                                                                   1,
                                                                   'weight_bn_act'),
                                 )

                layer += [edge]

            layer_list += [nn.ModuleList(layer)]

        return layer_list

    @staticmethod
    def build_reduction_layers(reduction_ops, normal_ops, input_channel, out_channel, num_layers):
        layer_list = []
        for num_edges in range(num_layers):
            layer = []
            for edge_idx in range(num_edges + 1):
                if edge_idx == 0:
                    ops = reduction_ops
                    in_C = input_channel
                    out_C = out_channel
                    S = 2
                else:
                    ops = normal_ops
                    in_C = out_channel
                    out_C = out_channel
                    S = 1

                edge = MixedEdge(candidate_ops=build_candidate_ops(ops,
                                                                   in_C,
                                                                   out_C,
                                                                   S,
                                                                   'weight_bn_act'),
                                 )

                layer += [edge]

                # for e in layer:
                #     print(e.AP_path_alpha)

            layer_list += [nn.ModuleList(layer)]

        return layer_list

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules
