# -*- coding: utf-8 -*-

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLossCosine(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_p = 1 - F.cosine_similarity(anchor, positive).view(-1, 1)
        d_n = 1 - F.cosine_similarity(anchor, negative).view(-1, 1)
        # p = 2
        # eps = 1e-6
        # d_p = F.pairwise_distance(anchor, positive, p, eps)
        # d_n = F.pairwise_distance(anchor, negative, p, eps)
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


def timer_with_task(job=""):
    def timer(fn):
        def wrapped(*args, **kw):
            print("{}".format(job + "..."))
            tic = time.time()
            ret = fn(*args, **kw)
            toc = time.time()
            print("{} Done. Time: {:.3f} sec".format(job, (toc - tic)))
            return ret

        return wrapped

    return timer
