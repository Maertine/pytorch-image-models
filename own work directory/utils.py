import torch
import torch.nn as nn

class RenyiDivergence(nn.Module):
    def __init__(self, alpha=0.5):
        super(RenyiDivergence, self).__init__()
        self.alpha = alpha

    def forward(self, input_logits, target_logits):
        q = torch.softmax(input_logits, dim=1)
        p = torch.softmax(target_logits, dim=1)

        if self.alpha == 1:  # KL Divergence
            return torch.sum(p * torch.log(p / q), dim=1).mean()

        return 1 / (self.alpha - 1) * torch.log(
            torch.sum(torch.pow(p, self.alpha) * torch.pow(q, 1 - self.alpha), dim=1)).mean()


class DKD(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(DKD, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_logits, target_logits, true_labels):
        target_mask = torch.zeros_like(target_logits).scatter_(1, true_labels.unsqueeze(1), 1).bool()

        p_t = torch.exp(torch.sum(target_logits * target_mask, dim=1, keepdim=True)) / torch.sum(
            torch.exp(target_logits), dim=1, keepdim=True)
        q_t = torch.exp(torch.sum(input_logits * target_mask, dim=1, keepdim=True)) / torch.sum(torch.exp(input_logits),
                                                                                                dim=1, keepdim=True)

        b = torch.cat([p_t, 1 - p_t], dim=1)
        d = torch.cat([q_t, 1 - q_t], dim=1)

        tckd = torch.sum(b * torch.log(b / d), dim=1)

        p_hat = F.softmax(target_logits - 10.0 * target_mask, dim=1)
        q_hat = F.softmax(input_logits - 10.0 * target_mask, dim=1)

        nckd = torch.sum(p_hat * torch.log(p_hat / q_hat), dim=1)

        # return torch.mean(tckd + (1-p_t) * nckd) ## If we plug this in we get KD

        return torch.mean(self.alpha * tckd + self.beta * nckd)