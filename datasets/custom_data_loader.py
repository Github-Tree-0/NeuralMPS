import torch.utils.data

def customDataloader(args):
    args.log.printWrite("=> fetching img pairs in %s" % (args.data_dir))
    datasets = __import__('datasets.' + args.dataset)
    dataset_file = getattr(datasets, args.dataset)
    train_set = getattr(dataset_file, args.dataset)(args, args.data_dir, 'train')
    val_set   = getattr(dataset_file, args.dataset)(args, args.data_dir, 'val')

    if args.concat_data:
        args.log.printWrite('****** Using cocnat data ******')
        args.log.printWrite("=> fetching img pairs in '{}'".format(args.data_dir2))
        train_set2 = getattr(dataset_file, args.dataset)(args, args.data_dir2, 'train')
        val_set2   = getattr(dataset_file, args.dataset)(args, args.data_dir2, 'val')
        train_set  = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set    = torch.utils.data.ConcatDataset([val_set,   val_set2])

    args.log.printWrite('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    args.log.printWrite('\t Train Batch: %d, Val Batch: %d' % (args.batch, args.val_batch))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return train_loader, test_loader

def my_MERL_Spectral_Dataloader(args):
    # Concat MERL together with rendered Spectral data

    Spectral_dir = '/userhome/kedaxiaoqiu_data/SDPS_npy'
    MERL_dir0 = 'data/datasets/PS_Blobby_Dataset'
    MERL_dir1 = 'data/datasets/PS_Sculpture_Dataset'

    Spectral_dataset = 'my_npy_dataloader'
    MERL_dataset = 'UPS_Synth_Dataset'

    args.log.printWrite("=> fetching img pairs in %s" % (MERL_dir0))
    MERL_datasets = __import__('datasets.' + MERL_dataset)
    MERL_dataset_file = getattr(MERL_datasets, MERL_dataset)
    train_set0 = getattr(MERL_dataset_file, MERL_dataset)(args, MERL_dir0, 'train')
    val_set0   = getattr(MERL_dataset_file, MERL_dataset)(args, MERL_dir0, 'val')

    args.log.printWrite("=> fetching img pairs in '{}'".format(MERL_dir1))
    train_set1 = getattr(MERL_dataset_file, MERL_dataset)(args, MERL_dir1, 'train')
    val_set1   = getattr(MERL_dataset_file, MERL_dataset)(args, MERL_dir1, 'val')

    args.log.printWrite("=> fetching img pairs in %s" % (Spectral_dir))
    Spectral_datasets = __import__('datasets.' + Spectral_dataset)
    Spectral_dataset_file = getattr(Spectral_datasets, Spectral_dataset)
    train_set2 = getattr(Spectral_dataset_file, Spectral_dataset)(args, Spectral_dir, 'train')
    val_set2 = getattr(Spectral_dataset_file, Spectral_dataset)(args, Spectral_dir, 'val')

    train_set  = torch.utils.data.ConcatDataset([train_set0, train_set1, train_set2])
    val_set    = torch.utils.data.ConcatDataset([val_set0, val_set1, val_set2])

    args.log.printWrite('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    args.log.printWrite('\t Train Batch: %d, Val Batch: %d' % (args.batch, args.val_batch))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return train_loader, test_loader


def benchmarkLoader(args):
    args.log.printWrite("=> fetching img pairs in 'data/%s'" % (args.benchmark))
    datasets = __import__('datasets.' + args.benchmark)
    dataset_file = getattr(datasets, args.benchmark)
    test_set = getattr(dataset_file, args.benchmark)(args, args.bm_dir, split='test')

    args.log.printWrite('Found Benchmark Data: %d samples' % (len(test_set)))
    args.log.printWrite('\t Test Batch %d' % (args.test_batch))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader
