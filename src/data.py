import os
import torch
import torchvision.datasets as dsets


class DataLoader:

    def __init__(self,X,y,batch_size):
        self.X, self.y = X, y 
        self.batch_size = batch_size
        self.n_samples = len(y)
        self.idx = 0

    def __len__(self):
        length = self.n_samples // self.batch_size
        if self.n_samples > length * self.batch_size:
            length += 1
        return length

    def __iter__(self):
        return self    

    def __next__(self):
        if self.idx >= self.n_samples:
            self.idx = 0
            rnd_idx = torch.randperm(self.n_samples)
            self.X = self.X[rnd_idx]
            self.y = self.y[rnd_idx]

        idx_end = min(self.idx+self.batch_size, self.n_samples)
        batch_X = self.X[self.idx:idx_end]
        batch_y = self.y[self.idx:idx_end]
        self.idx = idx_end

        return batch_X,batch_y


def load_fmnist(training_size, batch_size=100):
    train_set = dsets.FashionMNIST('data/fashionmnist', train=True, download=True)
    train_X, train_y = train_set.data[0:training_size].float()/255, \
                       to_one_hot(train_set.targets[0:training_size])
    train_loader = DataLoader(train_X, train_y, batch_size)

    test_set = dsets.FashionMNIST('data/fashionmnist', train=False,download=True)
    test_X, test_y = test_set.data.float()/255, \
                     to_one_hot(test_set.targets)
    test_loader = DataLoader(test_X, test_y, batch_size)

    return train_loader, test_loader


def load_cifar10(training_size, batch_size=100):
    """
    load cifar10 dataset. Notice that here we only use examples
    corresponding to label 0 and 1. Thus the training_size is at 
    most 10000.
    """
    train_set = dsets.CIFAR10('data/cifar10', train=True, download=True)
    train_X,train_y = modify_cifar_data(train_set.data, train_set.targets, training_size)
    train_loader = DataLoader(train_X, train_y, batch_size)

    test_set = dsets.CIFAR10('data/cifar10', train=False, download=True)
    test_X,test_y = modify_cifar_data(test_set.data, test_set.targets)
    test_loader = DataLoader(test_X, test_y, batch_size)

    return train_loader, test_loader


def modify_cifar_data(X, y, n_samples=-1):
    X = torch.from_numpy(X.transpose([0,3,1,2]))
    y = torch.LongTensor(y)

    X_t = torch.Tensor(50000,3,32,32)
    y_t = torch.LongTensor(50000)
    idx = 0
    for i in range(len(y)):
        if y[i] == 0 or y[i] == 1:
            y_t[idx] = y[i]
            X_t[idx,:,:,:] = X[i,:,:,:]
            idx += 1
    X = X_t[0:idx]
    y = y_t[0:idx] 

    if n_samples > 1:
        X = X[0:n_samples]
        y = y[0:n_samples]

    # preprocess the data
    X = X.float()/255.0
    y = to_one_hot(y) 

    return X, y


def to_one_hot(labels):
    if labels.ndimension()==1:
        labels.unsqueeze_(1)
    n_samples = labels.shape[0]
    n_classes = labels.max()+1

    one_hot_labels = torch.FloatTensor(n_samples,n_classes)
    one_hot_labels.zero_()
    one_hot_labels.scatter_(1, labels, 1)

    return one_hot_labels


if __name__ == '__main__':
    train_loader, test_loader = load_cifar10(training_size=10000,batch_size=500)
    for i in range(30):
        batch_x, batch_y = next(train_loader)
        print(i, batch_x.shape, batch_y.shape)

    for i in range(4):
        batch_x, batch_y = next(test_loader)
        print(i, batch_x.shape, batch_y.shape)


