# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse

from Recasting_ver.models import CifarRunConfig
from Recasting_ver.nas_recasting_manager import *
from Recasting_ver.models.super_nets.super_proxyless import SuperDartsRecastingNet
from Recasting_ver.models.model_zoo.model_zoo import *

# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {},
}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='cifar')
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='cifar10', choices=['imagenet', 'cifar10', 'cifar100'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--valid_size', type=int, default=0)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--num_blocks', type=str, default='4,4,4')
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

""" teacher config """
parser.add_argument('--teacher_blocks', type=str, default='9,9,9')
parser.add_argument('--teacher_type', type=str, default='resnet')
parser.add_argument('--teacher_filename', type=str, default='resnet56_pretrained.pth')
parser.add_argument('--teacher_path', type=str, default='cifar/resnet56')
parser.add_argument('--clear_pretrained_teacher', type=bool, default=False)
""" recasting config """
parser.add_argument('--teacher_matching_points', type=str, default=None)
parser.add_argument('--student_matching_points', type=str, default=None)
# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs', type=int, default=0)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['mobile', 'cpu', 'gpu8', 'flops', None])
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)
""" Debug config """
parser.add_argument('--print_arch_params', type=bool, default=False)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)
    os.makedirs(args.teacher_path, exist_ok=True)

    # build run config from args
    if args.dataset == 'cifar10' :
        args.n_classes = 10
    elif args.dataset == 'cifar100' :
        args.n_classes = 100
    elif args.dataset == 'imagenet' :
        args.n_classes = 1000
    else :
        raise NotImplementedError

    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = CifarRunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0

    # build net from args
    args.num_blocks = [int(val) for val in args.num_blocks.split(',')]
    args.normal_ops = [
                        '3x3_Conv', '5x5_Conv',
                        '3x3_ConvDW', '5x5_ConvDW',
                        '3x3_dConv', '5x5_dConv',
                        '3x3_dConvDW', '5x5_dConvDW',
                        '3x3_maxpool', '3x3_avgpool',
                       # 'Zero', 
                       # 'Identity',
                      ]
    args.reduction_ops = [
                        '3x3_Conv', '5x5_Conv',
                        '3x3_ConvDW', '5x5_ConvDW',
                        '3x3_dConv', '5x5_dConv',
                        '3x3_dConvDW', '5x5_dConvDW',
                        '2x2_maxpool', '2x2_avgpool',
                         ]
    # build teacher from args
    args.teacher_blocks = [int(val) for val in args.teacher_blocks.split(',')]
    # define teacher network
    teacher_net = build_resnet(num_blocks=args.teacher_blocks, num_classes=args.n_classes)
    # define super network
    super_net = SuperDartsRecastingNet(
        num_blocks=args.num_blocks, num_layers=args.num_layers, 
        normal_ops=args.normal_ops, reduction_ops=args.reduction_ops,
        n_classes=run_config.data_provider.n_classes,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
    )
    # recasting option
    if args.student_matching_points is None :
        s = 0
        args.student_idxs = []
        for num in args.num_blocks :
            sub = [i for i in range(s, s+num)]
            s += num
            args.student_idxs.append(sub)
    else :
        student_idxs = [int(val) for val in args.student_matching_points.split(',')]
    
        s = 0
        args.student_idxs = []
        for num in student_idxs :
            sub = [i for i in range(s, s+num)]
            s += num
            args.student_idxs.append(sub)
    
    if args.teacher_matching_points is None :
        s = 0
        args.teacher_idxs = []
        for num in args.teacher_blocks :
            sub = [i for i in range(s, s+num)]
            s += num
            args.teacher_idxs.append(sub)
    else :
        teacher_idxs = [int(val) for val in args.teacher_matching_points.split(',')]

        s = 0
        args.teacher_idxs = []
        for num in teacher_idxs :
            sub = [i for i in range(s, s+num)]
            s += num
            args.teacher_idxs.append(sub)

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    if args.arch_algo == 'grad':
        from Recasting_ver.nas_recasting_manager import GradientArchSearchConfig
        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        raise NotImplementedError
    else:
        raise NotImplementedError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    arch_search_run_manager = RecastingArchSearchRunManager(args.path, args.teacher_path, 
                                                            super_net, teacher_net, run_config, arch_search_config)

    # load pretrained teacher network
    #teacher_pretrained_file = os.path.join(args.path, args.teacher_pretrained)
    teacher_pretrained_file = os.path.join(args.teacher_path, 'checkpoint' ,args.teacher_filename)

    if args.clear_pretrained_teacher and os.path.exists(teacher_pretrained_file):
        os.remove(teacher_pretrained_file)

    if os.path.exists(teacher_pretrained_file):
        print('load pretrained teacher')
        arch_search_run_manager.load_teacher_model(model_fname=args.teacher_filename)
    else :
        print('train teacher network')
        arch_search_run_manager.train_teacher()
        print('save trained teacher network')
        arch_search_run_manager.save_teacher_model(model_fname=args.teacher_filename)
    
    print('Validate pretrained teacher network')
    arch_search_run_manager.validate()
    
    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            warmup_path = os.path.join(
                home, 'Workspace/Exp/arch_search/%s_DartsRecasting/warmup.pth.tar' %
                      (run_config.dataset)
            )
            if os.path.exists(warmup_path):
                print('load warmup weights')
                arch_search_run_manager.load_model(model_fname=warmup_path)
            else:
                print('fail to load models')

    print(arch_search_run_manager.teacher)
    print(arch_search_run_manager.net)
    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

    # recasting
    for s_idx, t_idx in zip(args.student_idxs, args.teacher_idxs):
        arch_search_run_manager.train_blocks(s_idx, t_idx)
    
    print('\n-----------------------------------------------------------------')
    print('Architecture search using recasting is finished')
    print('Start fine-tuning')
    print('-----------------------------------------------------------------')
    # joint training
    arch_search_run_manager.train_whole_network(fix_net_weights=args.debug)
