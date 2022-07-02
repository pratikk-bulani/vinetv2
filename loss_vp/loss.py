import torch, kldiv_loss

def get_loss(pred_map, gt):
    loss = torch.FloatTensor([0.0]).cuda()
    if True:
        loss += 1 * kldiv_loss.kldiv(pred_map, gt)
    return loss