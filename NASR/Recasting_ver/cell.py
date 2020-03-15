from Recasting_ver.models.normal_nets.proxyless_nets import DartsRecastingNet
from Recasting_ver.modules.layers import MyModule, nn
from Recasting_ver.modules.mix_op import MixedEdge, build_candidate_ops


class Cell(MyModule):
    """
        #TODO
        One Block == One Cell
    """

    def __init__(self, layer_list):
        super(Cell, self).__init__()

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, x):

        x_list = [x]
        for op_list in self.layer_list:
            x_out = op_list[0](x_list[0])
            for x_in, op in zip(x_list[1:], op_list[1:]):
                x_out = x_out + op(x_in)

            x_list += [x_out]

        return x_list[-1]

    @property
    def module_str(self):
        str_ = '['
        for op_list in self.layer_list:
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
        for op_list in self.layer_list:
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
        for op_list in self.layer_list:
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
        for op_list in self.layer_list:
            f, x_out = op_list[0].get_flops(x_list[0])
            flops += f
            for x_in, op in zip(x_list[1:], op_list[1:]):
                f_t, x_out_t = op.get_flops(x_in)
                f += f_t
                x_out += x_out_t
                flops += f

            x_list += [x_out]

        return flops, self.forward(x)




