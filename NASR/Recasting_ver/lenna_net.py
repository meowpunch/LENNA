from Recasting_ver.models.normal_nets.proxyless_nets import DartsRecastingNet, DartsRecastingBlock
from Recasting_ver.models.super_nets.super_proxyless import SuperDartsRecastingNet
from Recasting_ver.modules.layers import MyModule, nn, ConvLayer, LinearLayer
from Recasting_ver.modules.mix_op import MixedEdge, build_candidate_ops


class LennaNet(DartsRecastingNet):
    """
        Darts search structure
        first_conv -> Blocks(Cells) -> pool -> classifier
        TODO:
            we evaluate just one block(cell)
            LennaNet :
                first conv -> one block -> pool -> classifier
            we calculate latency of one block
    """

    def __init__(self, num_blocks, num_layers,
                 normal_ops, reduction_ops, block_type,
                 input_channel, output_channel,
                 n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0,
                 ):
        b_in_channel = input_channel
        b_out_channel = output_channel

        """
            first_conv.input_channel <- 3 (RGB)
            first_conv.output_channel <- b_input_channel 
        """
        first_conv = ConvLayer(
            3, b_in_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
        )

        if block_type:
            edges = self.build_normal_layers(normal_ops, b_in_channel, b_out_channel, num_layers)
            block = [DartsRecastingBlock(edges)]

        else:
            b_out_channel = b_in_channel * 2
            edges = self.build_reduction_layers(reduction_ops, normal_ops, b_in_channel, b_out_channel,
                                                num_layers)
            block = [DartsRecastingBlock(edges)]

        classifier = LinearLayer(b_out_channel, n_classes, dropout_rate=dropout_rate)
        super(LennaNet, self).__init__(first_conv, block, classifier)


    @staticmethod
    def build_normal_layers(candidate_ops, input_channel, out_channel, num_layers):
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
            layer_list += [nn.ModuleList(layer)]

        return layer_list
