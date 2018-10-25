import os
import time
import argparse
import json
import torch

from src.utils import load_net, load_data, \
                      get_sharpness, get_nonuniformity, \
                      eval_accuracy

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',default='0,')
    argparser.add_argument('--dataset',default='fashionmnist',
                            help='dataset choosed, [fashionmnist] | cifar10')
    argparser.add_argument('--n_samples',type=int,
                            default=1000, help='training set size, [1000]')
    argparser.add_argument('--batch_size', type=int,
                            default=1000, help='batch size')
    argparser.add_argument('--model_file', default='fnn.pkl',
                            help='file name of the pretrained model')
    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

    print('===> Config:')
    print(json.dumps(vars(args),indent=2))
    return args


def main():
    args = get_args()

    # load model
    criterion = torch.nn.MSELoss().cuda()
    train_loader,test_loader = load_data(args.dataset, 
                                        training_size=args.n_samples, 
                                        batch_size=args.batch_size)
    net = load_net(args.dataset)
    net.load_state_dict(torch.load(args.model_file))

    # Evaluate models
    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)

    print('===> Basic information of the given model: ')
    print('\t train loss: %.2e, acc: %.2f'%(train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f'%(test_loss, test_accuracy))

    print('===> Compute sharpness:')
    sharpness = get_sharpness(net, criterion, train_loader, \
                                n_iters=10, verbose=True, tol=1e-4)
    print('Sharpness is %.2e\n'%(sharpness))

    print('===> Compute non-uniformity:')
    non_uniformity = get_nonuniformity(net, criterion, train_loader, \
                                        n_iters=10, verbose=True, tol=1e-4)
    print('Non-uniformity is %.2e\n'%(non_uniformity))

if __name__ == '__main__':
    main()
