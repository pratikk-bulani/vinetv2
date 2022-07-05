import torch

def kldiv(s_map, gt):
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1)
    # expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    # assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1).view(batch_size, 1, 1)
    # expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    # assert expand_gt.size() == gt.size()

    s_map = s_map/(sum_s_map)
    gt = gt / (sum_gt)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt/(s_map + eps))
    return torch.mean(torch.sum(result, 1))

# import sys
# import torch
# from torch import nn

# def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
#     d = lambd(input, target)
#     if not reduce:
#         return d
#     return torch.mean(d) if size_average else torch.sum(d)

# class KLDLoss(nn.Module):
#     def __init__(self):
#         super(KLDLoss, self).__init__()

#     def KLD(self, inp, trg):
#         inp = inp/torch.sum(inp)
#         trg = trg/torch.sum(trg)
#         eps = sys.float_info.epsilon

#         return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

#     def forward(self, inp, trg):
#         return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)