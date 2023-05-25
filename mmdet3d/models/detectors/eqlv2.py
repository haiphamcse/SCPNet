import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
from functools import partial

def get_activation(cls_score):
    cls_score = torch.sigmoid(cls_score)
    n_i, n_c = cls_score.size()
    bg_score = cls_score[:, -1].view(n_i, 1)
    cls_score[:, :-1] *= (1 - bg_score)
    return cls_score

class EQLv2(nn.Module):
    def __init__(self,
                 num_classes=18,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self,
                cls_score,
                label):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        # pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        # neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
        neg_w = self.map_func(self.pos_neg)
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w
