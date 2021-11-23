import os
import torch
import torch.nn as nn
import argparse
import numpy as np

def getInput(args, data):
    input_list = [data['img']]
    if args.in_light: input_list.append(data['dirs_'])
    if args.in_mask:  input_list.append(data['m'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    ints = sample['ints']

    # dirs_ = sample['dirs'].expand_as(img) 已经放入LCNet
    dirs_ = sample['dirs'] # rescaling hasn't been done yet

    n, c, h, w = sample['dirs'].shape
    dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
    dirs = torch.cat(dirs_split, 0)

    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs_, dirs, ints = dirs_.cuda(), dirs.cuda(), ints.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'n': normal, 'm': mask, 'dirs': dirs, 'ints': ints, 'dirs_':dirs_}
    if args.random_ints:
        data['random_ints'] = sample['random_ints']
        if args.cuda:
            data['random_ints'] = data['random_ints'].cuda()
    return data 

def getInputChanel(args):
    args.log.printWrite('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        args.log.printWrite('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.in_mask:
        args.log.printWrite('[Network Input] Adding Mask as input')
        c_in += 1
    args.log.printWrite('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkp_{}.pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def conv_ReLU(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True)
                )

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def upconv(cin, cout):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def pytorch_variance_scaling_initializer(tensor, n, factor):
    "fan_in mode"
    std = np.sqrt(1.3 * factor / n)
    truncated_normal_(tensor, std=std)
    return None

def NR_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        shape = m.weight.data.shape
        n = shape[1] * shape[2] * shape[3]
        pytorch_variance_scaling_initializer(m.weight.data, n=n, factor=2.0)

def weight_init(tensor):
    n = tensor.shape[-2]
    gain = np.sqrt(2/(1+0.2**2))
    factor = gain * gain / 1.3
    pytorch_variance_scaling_initializer(tensor, n=n, factor=factor)

def GPS_init(tensor):
    shape = tensor.shape
    n = shape[1] * shape[2] * shape[3]
    gain = np.sqrt(2/(1+0.2**2))
    factor = gain * gain / 1.3
    pytorch_variance_scaling_initializer(tensor, n=n, factor=factor)

def model_init(model):
    model.NR_Net.apply(NR_init)
    weight_init(model.UFE.weights1_1)
    weight_init(model.UFE.weights1_2)
    weight_init(model.UFE.weights1_3)
    weight_init(model.UFE.weights2)
    GPS_init(model.conv.weight)

def gen_gps_args(ori_args):
    desc = "Args for gps model"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--num_cpu_cores', type=int, default=8)
    parser.add_argument('--image_h', type=int, default=ori_args.crop_h, help='The height of input images.')
    parser.add_argument('--image_w', type=int, default=ori_args.crop_w, help='The width of input images.')
    parser.add_argument('--NCHANNEL', type=int, default=6, help='The number of channels of each node.')
    parser.add_argument('--N', type=int, default=ori_args.input_num, help='The number of input images.')
    parser.add_argument('--NFILTER', type=int, default=32, help='The number of SGC Filters.')
    #if args.N<16, set args.NV = 32
    parser.add_argument('--NV', type=int, default=16)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--scope_idx', type=str, default='19')

    return parser.parse_args([])