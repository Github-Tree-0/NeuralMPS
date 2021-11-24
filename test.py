from numpy.compat.py3k import npy_load_module
import torch, sys

from datasets import custom_data_loader
from options  import run_model_opts
from models   import custom_model
from models import model_utils
from utils    import logger, recorders

import numpy as np
import matplotlib.pyplot as plt
import os

args = run_model_opts.RunModelOpts().parse()
args.stage2    = True
args.test_resc = False

args.retrain = 'checkpoints/stage1.pth.tar'
args.retrain_ps = 'checkpoints/stage2.pth.tar'
args.benchmark = 'my_npy_dataloader'
args.bm_dir = 'test_data/'
args.rescale = True
args.crop = False
args.color_aug = False
args.out_name = 'results'
args.input_num = 12

args.use_gt_ints = False
args.random_ints = False
log  = logger.Logger(args)

def main(args):
    out_dir = os.path.join(args.bm_dir, args.out_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    test_loader = custom_data_loader.benchmarkLoader(args)
    if args.retrain == 'None':
        model = None
    else:
        model = custom_model.buildModel(args)
    model_s2 = custom_model.build_PS(args)
    models = [model, model_s2]

    normal_maps = []
    objs = []

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            data = model_utils.parseData(args, sample)
            input = model_utils.getInput(args, data)
            
            if args.retrain == 'None':
                pred_c = {}
                if args.random_ints:
                    pred_c['intens'] = data['random_ints']
                else:
                    pred_c['intens'] = torch.ones(data['ints'].shape).cuda()
            else:
                pred_c = models[0](input)
                pred_c['intens'] = pred_c['intens'] * 20
            pred_c['dirs'] = data['dirs']

            if args.benchmark == 'my_npy_dataloader' or args.benchmark == 'my_npy_rgb_dataloader':
                shape, mat = sample['obj'][0].split('/')
                dirname = os.path.join(out_dir, shape)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                np.save(os.path.join(dirname, 'pred_intens_{}.npy'.format(mat)), pred_c['intens'].cpu().numpy())
                np.save(os.path.join(dirname, 'gt_intens_{}.npy'.format(mat)), data['ints'].cpu().numpy())
            else:
                obj = sample['obj'][0]
                np.save(os.path.join(out_dir, 'pred_intens_{}.npy'.format(obj)), pred_c['intens'].cpu().numpy())
                np.save(os.path.join(out_dir, 'gt_intens_{}.npy'.format(obj)), data['ints'].cpu().numpy())
            input.append(pred_c)
            pred = models[1](input)
            normal_map = (pred['n'].data + 1) / 2
            normal_map = normal_map * data['m'].data.expand_as(pred['n'].data)
            normal_maps.append(normal_map)
            if args.benchmark == 'my_npy_dataloader' or args.benchmark == 'my_npy_rgb_dataloader':
                objs.append([shape, mat])
            else:
                objs.append(obj)

    i = 0
    for normal_map, obj in zip(normal_maps, objs):
        np_normal = np.transpose(normal_map.cpu().numpy().squeeze(), (1, 2, 0))
        if args.benchmark == 'my_npy_dataloader' or args.benchmark == 'my_npy_rgb_dataloader':
            shape, mat = obj
            dirname = os.path.join(out_dir, shape)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            png_filename = os.path.join(dirname, 'Ours_NMPS_N_est_{}.png'.format(mat))
            npy_filename = os.path.join(dirname, 'Ours_NMPS_N_est_{}.npy'.format(mat))
        else:
            png_filename = os.path.join(out_dir, 'Ours_NMPS_N_est_{}.png'.format(obj))
            npy_filename = os.path.join(out_dir, 'Ours_NMPS_N_est_{}.npy'.format(obj))
        plt.imsave(png_filename, np_normal)
        np.save(npy_filename, np_normal)
        i += 1


if __name__ == '__main__':
    main(args)
