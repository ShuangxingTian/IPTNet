import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',              type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--n_colors',           type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--n_resblocks',        type=int, default=4,
                    help='number of residual blocks')
parser.add_argument('--n_resgroups',        type=int, default=1,
                    help='number of residual groups')
parser.add_argument('--istrain',            type=bool, default=False,
                    help='Model storage folder')
parser.add_argument('--istest',             type=bool, default=True,
                    help='Model storage folder')
# train
parser.add_argument('--save_path',          type=str, default='HEVC_IPTNet_X4cnn_QP22_20211210',
                    help='Model storage folder')
parser.add_argument('--train_low_path',     type=str, default=r"D:\tian\DataSets\interpolation\hevc\x4\qp22",
                    help='Model storage folder')
parser.add_argument('--train_high_path',    type=str, default=r"D:\tian\DataSets\interpolation\x4_gt",
                    help='Model storage folder')
parser.add_argument('--gt_patch_size',      type=tuple, default=(64, 64),
                    help='number of feature maps reduction')
parser.add_argument('--pre_batch_size',     type=int, default=6400,
                    help='number of feature maps reduction')
parser.add_argument('--batch_size',         type=int, default=64,
                    help='number of feature maps reduction')
parser.add_argument('--base_lr',            type=float, default=1e-3,
                    help='number of feature maps reduction')
parser.add_argument('--max_epoch',          type=int, default=500,
                    help='number of feature maps reduction')
parser.add_argument('--attenuation_epochs', type=int, default=20,
                    help='Model storage folder')
parser.add_argument('--resrme',             type=bool, default=False,
                    help='Model storage folder')
parser.add_argument('--save_minloss_model', type=bool, default=True,
                    help='Model storage folder')
parser.add_argument('--last_checkpoint',    type=str, default="./checkpoints/HEVC_IPTNet_X4cnn_QP22_20211209/HEVC_IPTNet_X4cnn_QP22_20211209_044.pth",
                    help='Model storage folder')
# test
parser.add_argument('--load_path',          type=str, default='HEVC_IPTNet_X4cnn_QP22_20211210',
                    help='Model storage folder')
parser.add_argument('--out_model_path',     type=str, default=r"D:\tian\HM-16.9-pt\model\IPTNet",
                    help='Model storage folder')
parser.add_argument('--low_path',           type=str, default=r"D:\tian\DataSets\interpolation\test\x4\low\qp22",
                    help='Model storage folder')
parser.add_argument('--gt_path',            type=str, default=r"D:\tian\DataSets\interpolation\test\x4\high",
                    help='Model storage folder')
parser.add_argument('--test_start_epoch',   type=int, default=0,
                    help='Model storage folder')

args = parser.parse_args()
