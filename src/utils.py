import math
from .models.vgg import vgg11
from .models.mnist import fnn
from .data import load_fmnist,load_cifar10
from .trainer import accuracy
from .linalg import eigen_variance, eigen_hessian



def load_net(dataset):
    if dataset == 'fashionmnist':
            return fnn().cuda()
    elif dataset == 'cifar10':
            return vgg11(num_classes=2).cuda()
    else:
        raise ValueError('Dataset %s is not supported'%(dataset))


def load_data(dataset, training_size, batch_size):
    if dataset == 'fashionmnist':
            return load_fmnist(training_size, batch_size)
    elif dataset == 'cifar10':
            return load_cifar10(training_size, batch_size)
    else:
        raise ValueError('Dataset %s is not supported'%(dataset))


def get_sharpness(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_hessian(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    return v


def get_nonuniformity(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_variance(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    return math.sqrt(v)


def eval_accuracy(model, criterion, dataloader):
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0

    loss_t, acc_t = 0.0, 0.0
    for i in range(n_batchs):
        inputs,targets = next(dataloader)
        inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)
        loss_t += criterion(logits,targets).item()
        acc_t += accuracy(logits.data,targets)

    return loss_t/n_batchs, acc_t/n_batchs
