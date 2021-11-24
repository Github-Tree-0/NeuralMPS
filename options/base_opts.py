import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Trainining Dataset ####
        self.parser.add_argument('--dataset',     default='my_npy_dataloader')
        self.parser.add_argument('--data_dir',    default='/userhome/kedaxiaoqiu_data/Blobby_npy')
        # self.parser.add_argument('--data_dir',    default='../ball_test')
        self.parser.add_argument('--data_dir2',   default='/userhome/kedaxiaoqiu_data/Sculpture_npy')
        self.parser.add_argument('--concat_data', default=True, action='store_false')
        self.parser.add_argument('--l_suffix',    default='_mtrl.txt')

        #### Training Data and Preprocessing Arguments ####
        self.parser.add_argument('--rescale',     default=False,  action='store_false')
        self.parser.add_argument('--rand_sc',     default=True,  action='store_false')
        self.parser.add_argument('--scale_h',     default=128,   type=int)
        self.parser.add_argument('--scale_w',     default=128,   type=int)
        self.parser.add_argument('--crop',        default=False,  action='store_false')
        self.parser.add_argument('--crop_h',      default=128,   type=int)
        self.parser.add_argument('--crop_w',      default=128,   type=int)
        self.parser.add_argument('--test_h',      default=128,   type=int)
        self.parser.add_argument('--test_w',      default=128,   type=int)
        self.parser.add_argument('--test_resc',   default=True,  action='store_false')
        self.parser.add_argument('--int_aug',     default=True,  action='store_false')
        self.parser.add_argument('--noise_aug',   default=True,  action='store_false')
        self.parser.add_argument('--noise',       default=0.05,  type=float)
        self.parser.add_argument('--color_aug',   default=True,  action='store_false')
        self.parser.add_argument('--color_ratio', default=3,     type=float)
        self.parser.add_argument('--normalize',   default=False, action='store_true')

        #### Device Arguments ####
        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--multi_gpu',   default=True, action='store_true')
        self.parser.add_argument('--time_sync',   default=True, action='store_true')
        self.parser.add_argument('--workers',     default=12,     type=int)
        self.parser.add_argument('--seed',        default=0,     type=int)

        #### Stage 1 Model Arguments ####
        self.parser.add_argument('--dirs_cls',    default=36,    type=int)
        self.parser.add_argument('--ints_cls',    default=15,    type=int)
        self.parser.add_argument('--dir_int',     default=False, action='store_true')
        self.parser.add_argument('--model',       default='LCNet')
        self.parser.add_argument('--fuse_type',   default='max')
        self.parser.add_argument('--in_img_num',  default=32,    type=int)
        self.parser.add_argument('--s1_est_n',    default=False, action='store_true')
        self.parser.add_argument('--s1_est_d',    default=False,  action='store_false')
        self.parser.add_argument('--s1_est_i',    default=True,  action='store_false')
        self.parser.add_argument('--in_light',    default=True, action='store_true')
        self.parser.add_argument('--in_mask',     default=True,  action='store_false')
        self.parser.add_argument('--use_BN',      default=False, action='store_true')
        self.parser.add_argument('--resume',      default=None)
        self.parser.add_argument('--retrain',     default='data/logdir/my_npy_dataloader/final_stage1_log/11-9,LCNet,max,adam,cos,ba_h-64,sc_h-128,cr_h-128,in_r-0.0005,no_w-1,di_w-1,in_w-1,in_m-32,di_s-36,in_s-15,in_light,in_mask,s1_est_i,color_aug,int_aug,concat_data/checkpointdir/checkp_25.pth.tar')
        self.parser.add_argument('--save_intv',   default=1,     type=int)
        self.parser.add_argument('--loss_miu',    default=5000,  type=float)

        #### Stage 2 Model Arguments ####
        self.parser.add_argument('--stage2',      default=False, action='store_true')
        self.parser.add_argument('--model_s2',    default='NENet')
        self.parser.add_argument('--retrain_s2',  default='data/logdir/my_npy_dataloader/final_12_gt_NE/11-9,LCNet,max,adam,NENet,ba_h-32,sc_h-128,cr_h-128,in_r-0.0005,no_w-1,di_w-1,in_w-1,in_m-32,di_s-36,in_s-15,in_light,in_mask,s1_est_i,color_aug,int_aug,concat_data,stage2/checkpointdir/checkp_17.pth.tar')
        #self.parser.add_argument('--retrain_s2',  default=None)
        self.parser.add_argument('--s2_est_n',    default=True,  action='store_false')
        self.parser.add_argument('--s2_est_i',    default=False, action='store_true')
        self.parser.add_argument('--s2_est_d',    default=False, action='store_true')
        self.parser.add_argument('--s2_in_light', default=True,  action='store_false')

        #### Displaying Arguments ####
        self.parser.add_argument('--train_disp',    default=20,  type=int)
        self.parser.add_argument('--train_save',    default=80, type=int)
        self.parser.add_argument('--val_intv',      default=1,   type=int)
        self.parser.add_argument('--val_disp',      default=1,   type=int)
        self.parser.add_argument('--val_save',      default=1,   type=int)
        self.parser.add_argument('--max_train_iter',default=-1,  type=int)
        self.parser.add_argument('--max_val_iter',  default=-1,  type=int)
        self.parser.add_argument('--max_test_iter', default=-1,  type=int)
        self.parser.add_argument('--train_save_n',  default=4,   type=int)
        self.parser.add_argument('--test_save_n',   default=4,   type=int)

        #### Log Arguments ####
        self.parser.add_argument('--save_root',  default='data/logdir/')
        self.parser.add_argument('--item',       default='CVPR2019')
        self.parser.add_argument('--suffix',     default=None)
        self.parser.add_argument('--debug',      default=False, action='store_true')
        self.parser.add_argument('--make_dir',   default=True,  action='store_false')
        self.parser.add_argument('--save_split', default=False, action='store_true')

        #### Additional options ####
        self.parser.add_argument('--retrain_ps',     default='data/models/PS-FCN_B_S_32.pth.tar')
        self.parser.add_argument('--retrain_gps',    default=None)
        self.parser.add_argument('--random_ints',    default=True, action='store_false')
        self.parser.add_argument('--input_num',      default=12,   type=int)
        self.parser.add_argument('--use_log_loss',   default=True, action='store_false')

    def setDefault(self):
        if self.args.debug:
            self.args.train_disp = 1
            self.args.train_save = 1
            self.args.max_train_iter = 4 
            self.args.max_val_iter = 4
            self.args.max_test_iter = 4
            self.args.test_intv = 1
    def collectInfo(self):
        self.args.str_keys  = [
                'model', 'fuse_type', 'solver'
                ]
        self.args.val_keys  = [
                'batch', 'scale_h', 'crop_h', 'init_lr', 'normal_w', 
                'dir_w', 'ints_w', 'in_img_num', 'dirs_cls', 'ints_cls'
                ]
        self.args.bool_keys = [
                'use_BN', 'in_light', 'in_mask', 's1_est_n', 's1_est_d', 's1_est_i', 
                'color_aug', 'int_aug', 'concat_data', 'retrain', 'resume', 'stage2', 
                ] 

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args

    def parse_jupyter(self):
        self.args = self.parser.parse_args([])
        return self.args
