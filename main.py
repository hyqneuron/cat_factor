from huva.th_util import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
from pprint import pprint, pformat
import argparse
import os

from model import *

"""
TODOs:
* test summing the categories to a single output
* Decaying neighbourhood
"""

"""
Categorical variables as factors, experiments:
1. mlpae: Plain autoencoder, with 5 factors each 5 categories as hidden code
"""

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name',  type=str, default='')
parser.add_argument('-m', '--mode',  type=str, default='mlpae')
parser.add_argument('-o', '--optimizer',  type=str, default='Adam')
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-f', '--nf', type=int, default=80)
parser.add_argument('-c', '--nc', type=int, default=10)
parser.add_argument('-cnnf', type=int, default=40)
parser.add_argument('-cnnc', type=int, default=10)
parser.add_argument('-wd','--weight-decay', type=float, default=0.0001)
parser.add_argument('--force-name', action='store_true')
args = parser.parse_args()

def make_data(batch_size):
    global dataset, dataset_test, loader, loader_test
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.126], std=[0.302])])
    def makeit(train):
        return torchvision.datasets.MNIST('/home/noid/data/torchvision_data/mnist', 
                                          train=train, transform=transforms, download=True)
    dataset      = makeit(True)
    dataset_test = makeit(False)
    loader       = DataLoader(dataset,      batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,num_workers=2, pin_memory=True)

def load_model(path):
    global model, model_name, model_path, args, other_args
    model = torch.load(path)
    model_name = model.model_name
    model_path = model.model_path
    args = model.args
    other_args = model.other_args

def make_all(name, batch_size=128, mode='mlpae', optmode='Adam', ignore_model=False):
    global model_name, model_path, logger, model, criterion, optimizer, wd, lr, other_args
    make_data(batch_size)
    assert mode.startswith('mlpae') or mode.startswith('cnnae')
    if ignore_model:
        assert 'model' in globals()
    else:
        nf = args.nf
        nc = args.nc
        cnnf = args.cnnf
        cnnc = args.cnnc
        # mlp variants
        if mode=='mlpae':
            model = make_mlpae(784, nf, nc)
        elif mode=='mlpae_s':
            model = make_mlpae_s(784, nf, nc)
        elif mode=='mlpae_d':
            model = make_mlpae_d(784, nf, nc)
        elif mode=='mlpae_ds':
            model = make_mlpae_ds(784, nf, nc)
        elif mode=='mlpae_bn':
            model = make_mlpae_bn(784, nf, nc)
        elif mode=='mlpae_bns':
            model = make_mlpae_bns(784, nf, nc)
        elif mode=='mlpae_simple':
            model = make_mlpae_simple(784, nf)
        # cnn variants
        elif mode=='cnnae':
            model = make_cnnae(1, cnnf, cnnc)
        elif mode=='cnnae_d':
            model = make_cnnae_d(1, cnnf, cnnc)
        elif mode=='cnnae_ds':
            model = make_cnnae_ds(1, cnnf, cnnc)
        elif mode=='cnnae_simple':
            model = make_cnnae_simple(1, cnnf)
        else:
            assert False, 'unknown mode'
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    wd = args.weight_decay
    if optmode=='Adam':
        lr = 0.001
        optimizer = MonitoredAdam(model.parameters(), lr, weight_decay=wd)
    elif optmode=='SGD':
        lr = 0.1
        optimizer = MonitoredSGD (model.parameters(), lr, weight_decay=wd)
    model_name = name
    model_path = 'logs/{}.pth'.format(model_name)
    logs_path  = 'logs/{}.log'.format(model_name)
    other_args = {'wd': wd, 'lr': lr}
    model.model_name = model_name
    model.model_path = model_path
    model.logs_path  = logs_path
    model.args       = args
    model.other_args = other_args
    model.printout   = str(model)
    if os.path.exists(logs_path):
        if not args.force_name:
            assert False, 'experiment log file {} exists. Try another name.'.format(logs_path)
    logger = LogPrinter(logs_path)
    logger.log(str(model))
    logger.log(pformat(args))
    logger.log(pformat(other_args))


epoch_trained = 0

def train(num_epochs, report_interval=40):
    global epoch_trained
    model.train()
    for epoch in xrange(num_epochs):
        gloss = 0
        for batch, (imgs, labels) in enumerate(loader):
            """ forward """
            if args.mode.startswith('mlp'):
                imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs.cuda()
            v_imgs = Variable(imgs)
            v_outs = model(v_imgs)
            v_loss = criterion(v_outs, v_imgs)
            """ backward """
            optimizer.zero_grad()
            v_loss.backward()
            optimizer.step()
            """ report """
            gloss += v_loss.data[0]
            if (batch+1) % report_interval == 0:
                avg_loss = gloss / report_interval
                gloss = 0
                logger.log('{:3d} {:3d} {:4d} {:6f} [{:6f} / {:6f}]'.format(
                    epoch_trained, epoch, batch, avg_loss, get_model_param_norm(model), optimizer.update_norm))

    epoch_trained += 1

def save_all():
    assert model_path is not None
    torch.save(model, model_path)

def decayed_training(schedule):
    global lr
    for epochs in schedule:
        train(epochs)
        lr /= 2
        set_learning_rate(optimizer, lr)
    logger.log(str(model))
    logger.log(pformat(args))
    logger.log(pformat(other_args))
    model.logtxt = logger.logtxt
    save_all()

def get_test_batches(max_batches=9999):
    batches = []
    for batch, (imgs, labels) in enumerate(loader_test):
        if batch >= max_batches: break
        if args.mode.startswith('mlp'):
            imgs = imgs.view(imgs.size(0), -1)
        batches.append(imgs)
    return batches

def make_stat(indices=[0,1]):
    global batches, name_layer, name_output, name_stat
    name_layer = {i:model[i] for i in indices}
    name_output = collect_output_over_loader(model, name_layer, loader_test, flatten=True)
    name_stat = {name: get_output_stats(output) for name, output in name_output.iteritems()}

def show_weight(weight, i, transpose=False, savepath=None):
    """
    weight is [Nout, Nin]
    visualize every row
    """
    if isinstance(weight, Variable):
        weight = weight.data.cpu()
    if transpose:
        weight = weight.transpose()
    if args.mode.startswith('mlpae'):
        assert weight.size(1) == 784
        w = weight[i].contiguous().view(28,28)
        plt.imshow(w.numpy())
    elif args.mode.startswith('cnnae'):
        assert weight.dim()==4 and weight.size(1)==1
        # [Nout, 1, 5, 5]
        w = weight[i, 0].squeeze() # [5,5]
        plt.imshow(w.numpy(), cmap='gray')
    else:
        assert False
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
        plt.close()

def vis(i):
    show_weight(model[0].weight, i)
    show_output_hist(name_outputstat[0][0], i)

def save_weights():
    for i in xrange(model[0].weight.size(0)):
        show_weight(model[0].weight, i, savepath='logs/{}_graphs/W0.{}.jpg'.format(model.model_name, i))

def t1(model_path='logs/mlpae4.pth'):
    global model 
    model = torch.load(model_path)
    make_data(128)
    make_stat()

if __name__=='__main__' and args.name != '':
    make_all(args.name, args.batch_size, mode=args.mode, optmode=args.optimizer)
    decayed_training([20])
    import os
    graphs_folder = 'logs/{}_graphs'.format(model_name)
    if not os.path.exists(graphs_folder):
        os.mkdir(graphs_folder)
    make_stat()
    save_weights()
