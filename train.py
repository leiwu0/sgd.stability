import os
import argparse
import json
import torch

from src.trainer import train
from src.utils import load_net, load_data, eval_accuracy


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',
                          default='0,', help='gpu id, [0] ')
    argparser.add_argument('--dataset', 
                          default='fashionmnist', help='dataset, [fashionmnist] | cifar10')
    argparser.add_argument('--n_samples', type=int,
                           default=1000, help='training set size, [1000]')
    argparser.add_argument('--load_size', type=int,
                           default=1000, help='load size for dataset, [1000]')
    argparser.add_argument('--optimizer', 
                           default='sgd', help='optimizer, [sgd]')
    argparser.add_argument('--n_iters', type=int,
                           default=10000, help='number of iteration used to train nets, [10000]')
    argparser.add_argument('--batch_size', type=int,
                           default=1000, help='batch size, [1000]')
    argparser.add_argument('--learning_rate', type=float,
                           default=1e-1, help='learning rate')
    argparser.add_argument('--momentum', type=float,
                           default='0.0', help='momentum, [0.0]')
    argparser.add_argument('--model_file', 
                           default='fnn.pkl', help='filename to save the net, fnn.pkl')

    args = argparser.parse_args()
    if args.load_size > args.batch_size:
        raise ValueError('load size should not be larger than batch size')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    print('===> Config:')
    print(json.dumps(vars(args), indent=2))
    return args

def get_optimizer(net, args):
    if args.optimizer == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('optimizer %s has not been supported'%(args.optimizer))

def main():
    args = get_args()

    criterion = torch.nn.MSELoss().cuda()
    train_loader, test_loader = load_data(args.dataset,
                                          training_size=args.n_samples,
                                          batch_size=args.load_size)
    net = load_net(args.dataset)
    optimizer = get_optimizer(net, args)
    print(optimizer)

    print('===> Architecture:')
    print(net)

    print('===> Start training')
    train(net, criterion, optimizer, train_loader, args.batch_size, args.n_iters, verbose=True)

    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
    print('===> Solution: ')
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    torch.save(net.state_dict(), args.model_file)


if __name__ == '__main__':
    main()
