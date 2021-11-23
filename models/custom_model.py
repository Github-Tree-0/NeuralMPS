from . import model_utils
import torch

def buildModel(args):
    print('Creating Model %s' % (args.model))
    in_c = model_utils.getInputChanel(args)
    other = {
            'img_num':  args.in_img_num, 
            'test_h':   args.test_h,   'test_w':   args.test_w,
            'in_mask':  args.in_mask,  'in_light': args.in_light, 
            'dirs_cls': args.dirs_cls, 'ints_cls': args.ints_cls,
            's1_est_d': args.s1_est_d, 's1_est_i': args.s1_est_i, 's1_est_n': args.s1_est_n, 
            }
    models = __import__('models.' + args.model)
    model_file = getattr(models, args.model)
    model = getattr(model_file, args.model)(args.fuse_type, args.use_BN, in_c, other)

    if args.cuda: model = model.cuda()

    if args.retrain: 
        args.log.printWrite("=> using pre-trained model '{}'".format(args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume:
        args.log.printWrite("=> Resume loading checkpoint '{}'".format(args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)
    print(model)
    args.log.printWrite("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model

def buildModelStage2(args):
    print('Creating Stage2 Model %s' % (args.model_s2))
    in_c = 6 if args.s2_in_light else 3
    other = {
            'img_num':  args.in_img_num,
            'in_mask':  args.in_mask,  'in_light': args.in_light, 
            'dirs_cls': args.dirs_cls, 'ints_cls': args.ints_cls,
            }
    models = __import__('models.' + args.model_s2)
    model_file = getattr(models, args.model_s2)
    model = getattr(model_file, args.model_s2)(args.fuse_type, args.use_BN, in_c, other)

    if args.cuda: model = model.cuda()

    if args.retrain_s2: 
        args.log.printWrite("=> using pre-trained model_s2 '{}'".format(args.retrain_s2))
        model_utils.loadCheckpoint(args.retrain_s2, model, cuda=args.cuda)

    print(model)
    args.log.printWrite("=> Stage2 Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model

def build_PS(args):
    print('Creating PS-FCN Model')
    in_c = 6
    other = {
            'in_light' : args.in_light,
            'in_mask'  : args.in_mask
            }
    model_type = 'PS_FCN'
    models = __import__('models.' + model_type)
    model_file = getattr(models, model_type)
    model = getattr(model_file, model_type)(args.fuse_type, args.use_BN, in_c, other)

    if args.cuda: model = model.cuda()

    if args.retrain_ps:
        args.log.printWrite("=> using pre-trained model_ps '{}'".format(args.retrain_ps))
        model_utils.loadCheckpoint(args.retrain_ps, model, cuda=args.cuda)
    
    print(model)
    args.log.printWrite("=> PS Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model    

def build_GPS(args):
    print('Creating GPS Model')
    gps_args = model_utils.gen_gps_args(args)
    model_type = 'GPS_Net'
    models = __import__('models.' + model_type)
    model_file = getattr(models, model_type)
    model = getattr(model_file, model_type)(gps_args)
    model_utils.model_init(model)
    if args.cuda: mdoel = model.cuda()

    if args.retrain_gps:
        args.log.printWrite("=> using pre-trained model_gps '{}'".format(args.retrain_gps))
        model_utils.loadCheckpoint(args.retrain_gps, model, cuda=args.cuda)
    
    print(model)
    args.log.printWrite("=> GPS Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model