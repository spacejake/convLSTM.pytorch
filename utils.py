import torch
from torch.autograd import Variable

def flip_var(var, axis):
    idx = [i for i in range(var.size(axis)-1, -1, -1)]
    idx = Variable(torch.LongTensor(idx))
    if var.is_cuda:
        idx = idx.cuda()
    inverted_tensor = var.index_select(axis, idx)
    return inverted_tensor

