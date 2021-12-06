import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale', default='2',
                    help='super resolution scale')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--n_resblocks', type=int, default=4,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=2,
                    help='residual scaling')
parser.add_argument('--n_resgroups', type=int, default=1,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
args = parser.parse_args()
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
