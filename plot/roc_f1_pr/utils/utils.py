import numpy as np

# compute accurate
def accuracy(output, target, topk=(1, 5)):
    # print(output.shape, target.shape)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# train
def adjust_learning_rate(args, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_anneal_schedule(t, nb_epoch, lr, optimizer):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    new_lr = float(lr / 2 * cos_out)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr 