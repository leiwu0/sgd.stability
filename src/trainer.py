import time
import torch

def train(model, criterion, optimizer, dataloader, batch_size, n_iters=50000, verbose=True):
    model.train()
    acc_avg, loss_avg = 0, 0

    since = time.time()
    for iter_now in range(n_iters):
        optimizer.zero_grad()
        loss,acc = compute_minibatch_gradient(model, criterion, dataloader, batch_size)
        optimizer.step()

        acc_avg = 0.9 * acc_avg + 0.1 * acc if acc_avg > 0 else acc
        loss_avg = 0.9 * loss_avg + 0.1 * loss if loss_avg > 0 else loss

        if iter_now%200 == 0 and verbose:
            now = time.time()
            print('%d/%d, took %.0f seconds, train_loss: %.1e, train_acc: %.2f'%(
                    iter_now+1, n_iters, now-since, loss_avg, acc_avg))
            since = time.time()


def compute_minibatch_gradient(model, criterion, dataloader, batch_size):
    loss,acc = 0,0
    n_loads = batch_size // dataloader.batch_size

    for i in range(n_loads):
        inputs,targets = next(dataloader)
        inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)
        E = criterion(logits,targets)
        E.backward()
        
        loss += E.item()
        acc += accuracy(logits.data,targets)

    for p in model.parameters():
        p.grad.data /= n_loads

    return loss/n_loads, acc/n_loads


def accuracy(logits, targets):
    n = logits.shape[0]
    if targets.ndimension() == 2:
        _, y_trues = torch.max(targets,1)
    else:
        y_trues = targets 
    _, y_preds = torch.max(logits,1)

    acc = (y_trues==y_preds).float().sum()*100.0/n 
    return acc












