import torch.nn as nn
import torch.nn.functional as funct


class KL_loss(nn.Module):
    def __init__(self, tau):
        super(KL_loss, self).__init__()
        self.tau = tau #temperatura

    def forward(self, y_s, y_t):
        p_s = funct.log_softmax(y_s / self.tau, dim=1)
        p_t = funct.softmax(y_t / self.tau, dim=1)
        kl_loss = funct.kl_div(p_s, p_t, reduction='batchmean') * (self.tau**2) #/ y_s.shape[0]
        return kl_loss
