from collections import deque

import torch


class LossMeter(object):
    """
    Measure loss and report average across iterations.
    """

    def __init__(self, average_over=20):
        self.average_over = average_over
        self.total_loss = 0.
        self.losses = deque(maxlen=self.average_over)

    def update(self, loss):
        self.total_loss += loss
        self.losses.append(loss)

    def __str__(self):
        avg_loss = sum(self.losses) / self.average_over
        return f"{avg_loss:06.1f}"


class SegMeter(object):
    """
    Score semantic segmentation metrics by accumulating histogram
    of outputs and targets.

    - overall pixel accuracy
    - per-class accuracy
    - per-class intersection-over-union (IU)
    - frequency weighted IU

    n.b. mean IU is the standard single number summary of segmentation quality.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = None

    def update(self, output, target):
        if self.hist is None:
            # init on first update (to inherit target device)
            self.hist = target.new_zeros((self.num_classes,) * 2).float()
        # metrics require hard assigment, so pick most likely class
        _, hard_seg = output.data[0].max(0)
        self.hist += self.fast_hist(target, hard_seg, self.num_classes)

    def score(self):
        scores = {}
        # overall accuracy
        scores['all_acc'] = torch.diag(self.hist).sum() / self.hist.sum()
        # per-class accuracy
        scores['acc'] = torch.diag(self.hist) / self.hist.sum(1)
        # per-class IU
        scores['iu'] = torch.diag(self.hist) / (self.hist.sum(1)
                                             + self.hist.sum(0)
                                             - torch.diag(self.hist))
        # frequency-weighted IU
        freq = self.hist.sum(1) / self.hist.sum()
        scores['freq_iu'] = (freq[freq > 0] * scores['iu'][freq > 0]).sum()
        return scores

    def fast_hist(self, a, b, n):
        """
        Fast 2D histogram by linearizing.
        """
        a, b = a.view(-1).long(), b.view(-1).long()
        k = (a >= 0) & (a < n)
        return torch.bincount(n * a[k] + b[k],
                              minlength=n**2).view(n, n).float()

    def __str__(self):
        strs = []
        for metric, score in self.score().items():
            score = torch.mean(score[score == score]).item()  # filter out NaN
            strs.append(f"{metric:10s} {score:.3f}")
        return '\n'.join(strs)
