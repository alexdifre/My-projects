import torch
from torch import nn
from CONTR_DISTILL.losses.memory import ContrastMemory 

eps = 1e-7

class CRDLoss(nn.Module):
    """CRD Loss function with separate embeddings for last and penultimate features"""
    def __init__(self, opt):
        super(CRDLoss, self).__init__()

        # Penultimate layer components
        self.embed_s_penult = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t_penult = Embed(opt.t_dim, opt.feat_dim)
        self.contrast_penult = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t_penult = ContrastLoss(opt.n_data)
        self.criterion_s_penult = ContrastLoss(opt.n_data)

        # Last layer components
        self.embed_s_last = Embed(opt.s_last_dim, opt.feat_dim)
        self.embed_t_last = Embed(opt.t_last_dim, opt.feat_dim)
        self.contrast_last = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t_last = ContrastLoss(opt.n_data)
        self.criterion_s_last = ContrastLoss(opt.n_data)

    def forward_penultimate(self, f_s, f_t, idx, contrast_idx=None):
        f_s = self.embed_s_penult(f_s)
        f_t = self.embed_t_penult(f_t)
        out_s, out_t = self.contrast_penult(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s_penult(out_s)
        t_loss = self.criterion_t_penult(out_t)
        return s_loss + t_loss

    def forward_last(self, f_s, f_t, idx, contrast_idx=None):
        f_s = self.embed_s_last(f_s)
        f_t = self.embed_t_last(f_t)
        out_s, out_t = self.contrast_last(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s_last(out_s)
        t_loss = self.criterion_t_last(out_t)
        return s_loss + t_loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.log(torch.div(P_pos, P_pos.add(m * Pn + eps)))

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x



class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
