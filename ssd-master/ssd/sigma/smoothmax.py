import torch


def smoothmax(x, y):
    """a softened, smooth maximum (not the softmax!)
    see https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    """
    min_ = torch.min(x, y)
    max_ = torch.max(x, y)
    return max_ + torch.log1p(torch.exp(min_ - max_))
