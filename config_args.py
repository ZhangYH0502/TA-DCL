import os.path as path 
import os
import numpy as np


def get_args(parser, eva=False):
    parser.add_argument('--dataroot', type=str, default='/research/deepeye/zhangyuh/data/')
    parser.add_argument('--dataset', type=str, choices=['coco', 'voc', 'chestmnist'], default='chestmnist')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--results_dir', type=str, default='results/')

    # Optimization
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=0.0001) # 0.00001
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--grad_ac_steps', type=int, default=1)
    parser.add_argument('--scheduler_step', type=int, default=1000)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--int_loss', type=float, default=0.0)
    parser.add_argument('--aux_loss', type=float, default=0.0)
    parser.add_argument('--loss_type', type=str, choices=['bce', 'mixed', 'class_ce', 'soft_margin'], default='bce')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'step'], default='plateau')
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--max_batches', type=int, default=-1)

    # Model
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pos_emb', action='store_true', help='positional encoding')
    parser.add_argument('--use_lmt', dest='use_lmt', action='store_true', default=False, help='label mask training')
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    
    # Image Sizes
    parser.add_argument('--scale_size', type=int, default=640) #640, 28
    parser.add_argument('--crop_size', type=int, default=576)

    # Testing Models
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--saved_model_name', type=str, default='logs/')
    
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--name', type=str, default='')
    args = parser.parse_args()

    model_name = args.dataset
    if args.dataset == 'voc':
        args.num_labels = 20
    elif args.dataset == 'coco':
        args.num_labels = 80
    elif args.dataset == 'chestmnist':
        args.num_labels = 14
    else:
        print('dataset not included')
        exit()

    model_name += '.'+str(args.layers)+'layer'
    model_name += '.bsz_{}'.format(int(args.batch_size * args.grad_ac_steps))
    model_name += '.'+args.optim+str(args.lr) #.split('.')[1]

    if args.pos_emb:
        model_name += '.pos_emb'

    if args.int_loss != 0.0:
        model_name += '.int_loss'+str(args.int_loss).split('.')[1]

    if args.aux_loss != 0.0:
        model_name += '.aux_loss'+str(args.aux_loss).replace('.', '')

    if args.name != '':
        model_name += '.'+args.name
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    model_name = os.path.join(args.results_dir, model_name)
    
    args.model_name = model_name

    if args.inference:
        args.epochs = 1

    if os.path.exists(args.model_name) and (not args.overwrite) and (not 'test' in args.name) and (not eva) and (not args.inference) and (not args.resume):
        print(args.model_name)
        # overwrite_status = input('Already Exists. Overwrite?: ')
        # if overwrite_status == 'rm':
        #     os.system('rm -rf '+args.model_name)
        # elif not 'y' in overwrite_status:
        #     exit(0)
    elif not os.path.exists(args.model_name):
        os.makedirs(args.model_name)

    return args
