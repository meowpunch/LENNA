from Recasting_ver.models.normal_nets.proxyless_nets import DartsRecastingNet, DartsRecastingBlock
from Recasting_ver.modules.layers import MyModule, nn, ConvLayer, LinearLayer, ZeroLayer
from Recasting_ver.modules.mix_op import MixedEdge, MixedEdge_v2, build_candidate_ops
from queue import Queue
import copy
import numpy as np
import torch

class LennaNet(DartsRecastingNet):
    def __init__(self, num_blocks, num_layers,
                 normal_ops, reduction_ops, block_type,
                 input_channel, n_classes=1000,
                 mixedge_ver=2, threshold=0.5,
                 increase_option=False, increase_term=10,
                 bn_param=(0.1, 1e-3), dropout_rate=0):
        self._redundant_modules = None
        self._unused_modules = None
        self.num_layers = num_layers
        self.mixedge_ver = mixedge_ver
        self.threshold = threshold
        self.increase_option = increase_option
        self.increase_term = increase_term

        self.fsize = list()

        # input_channel = make_divisible(16, 8) #returns 16
        f = 32

        # first conv layer
        self.fsize.append(f)
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=1, use_bn=True, act_func='relu', ops_order='weight_bn_act'
        )

        # blocks
        blocks = []
        total_blocks = sum(num_blocks)
        output_channel = input_channel

        for i, nb in enumerate(num_blocks):
            input_channel = output_channel

            if block_type:  # normal block
                edges, fsize = self.build_normal_layers(normal_ops, input_channel, output_channel, self.num_layers, f)
                b = DartsRecastingBlock(edges)
                blocks += [b]
                self.fsize += [fsize]
            else:  # Reduction blocks
                output_channel = input_channel * 2
                edges, fsize = self.build_reduction_layers(reduction_ops, normal_ops, input_channel, output_channel,
                                                           self.num_layers, f)
                b = DartsRecastingBlock(edges)
                blocks += [b]
                self.fsize += [fsize]
                f //= 2

            for _ in range(nb - 1):  # Normal blocks (will not be calculated)
                # fsize_list.append(f)
                edges, fsize = self.build_normal_layers(normal_ops, output_channel, output_channel, self.num_layers, f)
                b = DartsRecastingBlock(edges)
                blocks += [b]
                self.fsize += [fsize]

        classifier = LinearLayer(output_channel, n_classes, dropout_rate=dropout_rate)
        self.fsize.append(f)
        super(LennaNet, self).__init__(first_conv, blocks, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def arch_params_prob(self):
        """
            block -> layer -> mixed_edge
        """
        prob_list = []
        for block in self.blocks:
            for layer in block.layer_list:
                for mixed_edge in layer:
                    prob_list.append(mixed_edge.probs_over_ops)
        return prob_list

    def architecture_parameters_recasting(self, block_idxs):
        for idx in block_idxs:
            for name, param in self.blocks[idx].named_parameters():
                if 'AP_path_alpha' in name:
                    yield param

    def zero_parameters_recasting(self, block_idxs):
        for idx in block_idxs:
            for name, param in self.blocks[idx].named_parameters():
                if 'AP_zero_alpha' in name:
                    yield param

    def total_architecture_parameters_recasting(self, block_idxs):
        for idx in block_idxs:
            for name, param in self.blocks[idx].named_parameters():
                if 'AP_path_alpha' in name or 'AP_zero_alpha' in name:
                    yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name and 'AP_zero_alpha' not in name:
                yield param

    def initialize_zero(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge_v2'):
                    module._modules[m].reset_zero_alpha()
                    module._modules[m].reset_sigmoid_alpha()
                    module._modules[m].reset_hard_zero()
                    module._modules[m].reset_skip_zero()
                else:
                    queue.put(child)

    def zero_alpha_step(self, epoch, n_epochs):

        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge_v2'):
                    if self.increase_option == 1:
                        if epoch > n_epochs / 2:
                            module._modules[m].compute_zero()
                        if epoch > n_epochs / 2 and epoch % self.increase_term == 0:
                            module._modules[m].increase_sigmoid_alpha()
                        if epoch > n_epochs / 8 * 7:
                            module._modules[m].set_hard_zero()
                    elif self.increase_option == 2:
                        module._modules[m].increase_sigmoid_alpha_cont(0.05)
                else:
                    queue.put(child)

    def train_zero(self):

        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge_v2'):
                    module._modules[m].compute_zero()
                else:
                    queue.put(child)

    """ architecture parameters related methods """

    def build_normal_layers(self, candidate_ops, input_channel, out_channel, num_layers, f):
        layer_list = []
        f_list = []
        for num_edges in range(num_layers):
            layer = []
            f_sub_list = []
            for _ in range(num_edges + 1):
                if self.mixedge_ver == 1:
                    edge = MixedEdge(candidate_ops=build_candidate_ops(candidate_ops,
                                                                       input_channel,
                                                                       out_channel,
                                                                       1,
                                                                       'weight_bn_act'))
                elif self.mixedge_ver == 2:
                    edge = MixedEdge_v2(candidate_ops=build_candidate_ops(candidate_ops,
                                                                          input_channel,
                                                                          out_channel,
                                                                          1,
                                                                          'weight_bn_act'),
                                        threshold=self.threshold)
                else:
                    raise NotImplementedError

                layer += [edge]
                f_sub_list += [f]
            layer_list += [nn.ModuleList(layer)]
            f_list += [f_sub_list]

        return layer_list, f_list

    def build_reduction_layers(self, reduction_ops, normal_ops, input_channel, out_channel, num_layers, f):
        layer_list = []
        f_list = []
        for num_edges in range(num_layers):
            layer = []
            f_sub_list = []
            for edge_idx in range(num_edges + 1):
                if edge_idx == 0:
                    ops = reduction_ops
                    in_C = input_channel
                    out_C = out_channel
                    S = 2
                    f_sub_list += [f]
                else:
                    ops = normal_ops
                    in_C = out_channel
                    out_C = out_channel
                    S = 1
                    f_sub_list += [f // 2]

                if self.mixedge_ver == 1:
                    edge = MixedEdge(candidate_ops=build_candidate_ops(ops,
                                                                       in_C,
                                                                       out_C,
                                                                       S,
                                                                       'weight_bn_act'))
                elif self.mixedge_ver == 2:
                    edge = MixedEdge_v2(candidate_ops=build_candidate_ops(ops,
                                                                          in_C,
                                                                          out_C,
                                                                          S,
                                                                          'weight_bn_act'),
                                        threshold=self.threshold)
                else:
                    raise NotImplementedError

                layer += [edge]
            layer_list += [nn.ModuleList(layer)]
            f_list += [f_sub_list]

        return layer_list, f_list

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        # self.logger.info("init_ratio: {}".format(init_ratio))
        # for param in self.architecture_parameters():
        #     param_len = param.size()[0]
        #     param.data = torch.from_numpy(np.random.multinomial(1000, [1/param_len]*param_len)).float()

        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                # param.data.uniform_(-init_ratio, init_ratio)
                param.data.uniform_(init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if self.mixedge_ver == 1 and MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            elif self.mixedge_ver == 2 and MixedEdge_v2.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, LennaNet)  # SuperDartsRecastingNet originally
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    # def get_latency(self, latency_model: LatencyEstimator):
    #     latency = 0
    #
    #     # first conv
    #     latency += self.first_conv.get_latency(self.fsize[0], latency_model)
    #
    #     # classifier
    #     latency += self.classifier.get_latency(0, latency_model)
    #
    #     for i in range(len(self.blocks)):
    #         block = self.blocks[i]
    #         latency += block.get_latency(self.fsize[i + 1], latency_model)
    #
    #     return latency
    #
    # def expected_latency(self, latency_model: LatencyEstimator, block_idx):
    #     expected_latency = 0
    #
    #     for i in block_idx:
    #         block = self.blocks[i]
    #         expected_latency += block.expected_latency(self.fsize[i + 1], latency_model)
    #
    #     return expected_latency

    def expected_flops(self, x, block_idx):
        expected_flops = 0
        # first conv
        _, x = self.first_conv.get_flops(x)

        for i in range(len(self.blocks)):
            if i in block_idx:
                flops, x = self.blocks[i].get_flops(x)
                expected_flops += flops
            else:
                _, x = self.blocks[i].get_flops(x)

        return expected_flops

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return DartsRecastingNet(self.first_conv, list(self.blocks), self.classifier)

    def convert_to_normal_net_recasting(self, block_idx):
        queue = Queue()
        for i in block_idx:
            queue.put(self.blocks[i])
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)

    def reduce_ops(self, block_idx):
        queue = Queue()
        for i in block_idx:
            queue.put(self.blocks[i])
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    child.reduce_op()
                    child.cuda()
                else:
                    queue.put(child)

    def remove_unused_op(self, block_idx):
        for i in block_idx:
            l_list = self.blocks[i].layer_list

            # forward removal

            for i in range(len(l_list)):
                ll = l_list[i]
                remove = True
                for j in range(len(ll)):
                    if not isinstance(ll[j], ZeroLayer):
                        remove = False
                        break
                if remove is True:
                    for ii in range(i + 1, len(l_list)):
                        l_list[ii][i + 1] = ZeroLayer()

            # backward removal

            n_rows = len(l_list)
            n_cols = len(l_list[-1])

            for j in reversed(range(n_cols)):
                remove = True
                for i in reversed(range(n_rows)):
                    ll = l_list[i]
                    if j < len(ll) and isinstance(ll[j], ZeroLayer) is False:
                        remove = False
                        break

                if remove is True:
                    for ii in range(len(l_list[j - 1])):
                        l_list[j - 1][ii] = ZeroLayer()
