import torch
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared, DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import cupy as np
from scipy.ndimage import distance_transform_edt
import time as time

def compute_edts_forPenalizedLoss(GT, smooth=1e-8):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask
        
        res[i] = pos_edt/(np.max(pos_edt) + smooth) + neg_edt/(np.max(neg_edt) + smooth)
    return res

def get_dist_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return: Dice score scaled by voxel-wise distance maps for each class
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape
    num_classes = net_output.shape[1]

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    for c in range(num_classes):
        # print(f"Iteration for class #{c}")
        # start = time.time()
        gt_c = y_onehot[:,c,...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_c.numpy()>0.5) + 1.0
        dist = torch.from_numpy(dist)
        if net_output.device.type == "cuda":
            dist = dist.cuda(net_output.device.index)
        # end = time.time()
        # print(f"Time for distance mapping: {end-start} seconds")

        tp[:,c,...] = tp[:,c,...] * dist
        # fp[:,c,...] = fp[:,c,...] * dist
        # fn[:,c,...] = fn[:,c,...] * dist

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return dist_tp, fp, fn, tn

class DistDiceLoss(SoftDiceLoss):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5):
        """
        Distance map penalized Dice loss for multiclass segmentation
        Motivated by: https://openreview.net/forum?id=B1eIcvS45V
        Distance Map Loss Penalty Term for Semantic Segmentation
        Adapted from the Binary version: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py
        """
        super(DistDiceLoss, self).__init__(apply_nonlin, batch_dice, False, smooth)

    def forward(self, x, y, loss_mask=None):
        start = time.time()
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dist_tp, fp, fn, _ = get_dist_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * dist_tp + self.smooth
        denominator = 2 * dist_tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        end = time.time()
        print(f"Dist Dice Loss time: {end-start} seconds")
        return -dc

class DistDC_and_CE_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Distance-Mapped Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice,
                         log_dice, ignore_label)

        self.dc = DistDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        start = time.time()
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        end = time.time()
        print(f"Dist Dice Score: {end-start}")
        return result

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Binary Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation
    Taken from: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt, loss_mask=None):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        print(f"GT after one-hot: {y_onehot.size()}")
        
        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)
        print(f"Distances: {dist.size()}")

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
        
        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))
        
        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)

        dc = dc.mean()

        return -dc