# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F

from Recasting_ver.modules.layers import *


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        # 'Identity': lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Identity': lambda in_C, out_C, S: MyIdentity(),
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    ## Modifier : shorm21
    # add Mixed layers
    name2ops.update({
        '3x3_Conv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 3, S, 1),
        '5x5_Conv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 5, S, 1),
        #######################################################################################
        '3x3_ConvDW': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 3, S, 1),
        '5x5_ConvDW': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 5, S, 1),
        #######################################################################################
        '3x3_dConv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 3, S, 2),
        '5x5_dConv': lambda in_C, out_C, S: ConvLayer(in_C, out_C, 5, S, 2),
        #######################################################################################
        '3x3_dConvDW': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 3, S, 2),
        '5x5_dConvDW': lambda in_C, out_C, S: DepthConvLayer(in_C, out_C, 5, S, 2),
        #######################################################################################
        '2x2_maxpool': lambda in_C, out_C, S: PoolingExpandLayer(in_C, out_C, 'max', 2, S),
        '2x2_avgpool': lambda in_C, out_C, S: PoolingExpandLayer(in_C, out_C, 'avg', 2, S),
        '3x3_maxpool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'max', 3, S),
        '3x3_avgpool': lambda in_C, out_C, S: PoolingLayer(in_C, out_C, 'avg', 3, S),
        #######################################################################################
        '3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
        '3x3_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 2),
        '3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
        '3x3_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 4),
        '3x3_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 5),
        '3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
        #######################################################################################
        '5x5_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 1),
        '5x5_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 2),
        '5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
        '5x5_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 4),
        '5x5_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 5),
        '5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
        #######################################################################################
        '7x7_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 1),
        '7x7_MBConv2': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 2),
        '7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
        '7x7_MBConv4': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 4),
        '7x7_MBConv5': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 5),
        '7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
    })

    return [
        name2ops[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


class MixedEdge(MyModule):
    MODE = None  # full, two, None, full_v2

    # MODE = 'full_v2'

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    def get_active_alpha(self):
        return torch.sum(self.AP_path_alpha * self.AP_path_wb)

    def get_alpha(self):
        return self.AP_path_alpha

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    """ """

    def forward(self, x):
        if MixedEdge.MODE == 'full' or MixedEdge.MODE == 'two':
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi
            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi.detach()
        elif MixedEdge.MODE == 'full_v2':
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)

                return forward

            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                out_k = candidate_ops[k](_x.data)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads

                return backward

            output = ArchGradientFunction.apply(
                x, self.AP_path_wb, run_function(self.candidate_ops, self.active_index[0]),
                backward_function(self.candidate_ops, self.active_index[0], self.AP_path_wb)
            )
        else:
            output = self.active_op(x)
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def arch_params(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        return probs

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    """ """

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if MixedEdge.MODE == 'two':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i]
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        if self.AP_path_wb.grad is None:
            return
        binary_grads = self.AP_path_wb.grad.data
        #        if self.active_op.is_zero_layer() or isinstance(self.active_op, MyIdentity):
        #            self.AP_path_alpha.grad = None
        #            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedEdge.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset


class MixedEdge_v2(MyModule):
    MODE = None  # full, two, None, full_v2

    # MODE = 'full_v2'

    def __init__(self, candidate_ops, threshold=0.5):
        super(MixedEdge_v2, self).__init__()

        self.zero = None
        for x in candidate_ops:
            if isinstance(x, ZeroLayer):
                self.zero = x
                candidate_ops.remove(x)

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.AP_zero_alpha = Parameter(torch.Tensor(1).fill_(0))  # zero architecture parameters

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

        self.threshold = threshold
        self.skip_zero = True
        self.sigmoid_alpha = 1
        self.hard_zero = False

    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        # 0~1 에서 뽑고 normalize
        # 편차 크도록

        param_len = self.AP_path_alpha.size()[0]

        # 01
        # arr = []
        # arr.append(np.random.uniform(0, 1))
        # for i in range(param_len - 1):
        #     arr.append(np.random.uniform(0, 1) * (1 - sum(arr)))
        #
        # probs = np.array(arr)
        # np.random.shuffle(probs)
        # probs = torch.from_numpy(probs)

        # 02
        arr2 = list(np.random.uniform(0, 1, param_len))
        sum_arr = sum(arr2)
        probs = torch.from_numpy(np.array(map(lambda x: x / sum_arr, arr2)))

        # dices = 50
        # probs = torch.from_numpy(np.random.multinomial(dices, [1/param_len]*param_len)/dices)

        # 다 같은 값
        # probs = torch.from_numpy(np.array([1/param_len]*param_len))
        return probs

        # probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        # return probs

    @property
    def prob_zero(self):
        if self.hard_zero is False:
            prob = torch.sigmoid(self.sigmoid_alpha * self.AP_zero_alpha)
        else:
            #            if self.AP_zero_alpha.data >= 0 :
            #                prob = 1.0
            #            else :
            #                prob = 0.0
            temp_p = torch.sigmoid(self.sigmoid_alpha * self.AP_zero_alpha)
            if temp_p >= self.threshold:
                prob = 1.0
            else:
                prob = 0.0
        return prob

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        if self.AP_zero_alpha.data >= 0 or self.zero is None:
            return self.candidate_ops[index]
        else:
            return self.zero

    @property
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.AP_zero_wb.data

    def reset_sigmoid_alpha(self):
        self.sigmoid_alpha = 1

    def reset_zero_alpha(self):
        self.AP_zero_alpha.data.fill_(0)

    def reset_hard_zero(self):
        self.hard_zero = False

    def reset_skip_zero(self):
        if self.zero is not None:
            self.skip_zero = True

    def increase_sigmoid_alpha(self):
        self.sigmoid_alpha *= 2

    def increase_sigmoid_alpha_cont(self, coeff):
        self.sigmoid_alpha *= 2 ** coeff

    def compute_zero(self):
        if self.zero is not None:
            self.skip_zero = False

    def set_hard_zero(self):
        if self.zero is not None:
            self.hard_zero = True

    def get_active_alpha(self):
        return torch.sum(self.AP_path_alpha * self.AP_path_wb)

    def get_alpha(self):
        return self.AP_path_alpha

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def reduce_op(self):
        chosen_index, probs = self.chosen_index

        self.candidate_ops = nn.ModuleList([self.candidate_ops[chosen_index]])
        self.AP_path_alpha = Parameter(torch.Tensor(1).fill_(self.AP_path_alpha.data[chosen_index]))
        self.AP_path_wb = Parameter(torch.Tensor(1).fill_(self.AP_path_wb.data[chosen_index]))

    def forward(self, x):
        if MixedEdge_v2.MODE == 'full' or MixedEdge_v2.MODE == 'two':
            output = 0
            for _i in self.active_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi
            for _i in self.inactive_index:
                oi = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * oi.detach()
        elif MixedEdge_v2.MODE == 'full_v2' and self.hard_zero is False:
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)

                return forward

            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                out_k = candidate_ops[k](_x.data)
                            else:
                                out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads

                return backward

            output = ArchGradientFunction.apply(
                x, self.AP_path_wb, run_function(self.candidate_ops, self.active_index[0]),
                backward_function(self.candidate_ops, self.active_index[0], self.AP_path_wb)
            )
        else:
            output = self.active_op(x)

        if self.zero is None:
            output = 1 * output
        elif self.zero is not None and self.skip_zero is False:
            output = self.prob_zero * output
        else:
            output = 0.5 * output

        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix_v2(%s, %.3f, Connected, %.3f)' % (
        self.candidate_ops[chosen_index].module_str, probs, self.prob_zero)

    @property
    def arch_params(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        return probs

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    """ """

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if len(self.candidate_ops) <= 1:
            sample = 0
            self.active_index = [sample]
            self.inactive_index = []
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs[sample]
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0
        elif MixedEdge_v2.MODE == 'two':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i]
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample + 1, self.n_choices)]
            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0
        # avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):
        if self.AP_path_wb.grad is None:
            return
        binary_grads = self.AP_path_wb.grad.data
        if isinstance(self.active_op, MyIdentity):
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedEdge_v2.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset

    def expected_latency(self, fsize, latency_model):

        latency = 0
        if self.prob_zero == 0:
            return latency

        for op, p in zip(self.candidate_ops, self.probs_over_ops):
            if isinstance(op, MyIdentity):
                continue
            else:
                config = op.config
                name = op.module_str
                stride = config['stride']
                in_channels = op.in_channels
                out_channels = op.out_channels

                latency += p * latency_model.predict(name, fsize, stride, in_channels, out_channels)

        return latency

    def get_latency(self, fsize, latency_model):
        latency = 0

        if self.prob_zero == 0:
            return latency

        for i in self.active_index:
            latency += self.candidate_ops[i].get_latency(fsize, latency_model)
        return latency


class ArchGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x)
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None
