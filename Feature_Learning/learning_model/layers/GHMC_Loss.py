import torch
import torch.nn.functional as F


class GHMC_Loss:
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)
         
        temp = torch.zeros(input.size(0), input.size(1))
        target_onehot = temp.scatter_(1, target.cpu().view(-1, 1), 1).cuda()
        # gradient length
        g = torch.abs(input.sigmoid().detach() - target_onehot)

        #valid = mask > 0
        #tot = max(valid.float().sum().item(), 1.0)
        tot = input.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) #& valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target_onehot, weights, reduction='sum') / tot
        return loss