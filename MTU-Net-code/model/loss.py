import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
import cv2

def FocalIoULoss(inputs, targets):
    "Non weighted version of Focal Loss"

    # def __init__(self, alpha=.25, gamma=2):
    #     super(WeightedFocalLoss, self).__init__()
    # targets =
    # inputs = torch.relu(inputs)
    [b,c,h,w] = inputs.size()

    inputs = torch.nn.Sigmoid()(inputs)
    inputs = 0.999*(inputs-0.5)+0.5
    BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
    intersection = torch.mul(inputs, targets)
    smooth = 1

    # IoU = (intersection.sum(dim = [1,2,3]) + smooth) / (inputs.sum(dim = [1,2,3]) + targets.sum(dim = [1,2,3]) - intersection.sum(dim = [1,2,3]) + smooth)
    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)
    # IoU = IoU*0.5+0.5




    # inputs = torch.nn.Sigmoid()(inputs)

    # inputs = torch.nn.Hardsigmoid()(inputs)

    # inputs = torch.tanh(inputs)





    # intersection_sum = intersection.sum()

    # print(IoU)
    # IoU = IoU.unsqueeze(-1)
    # IoU = IoU.unsqueeze(-1)
    # IoU = IoU.unsqueeze(-1)
    # IoU = IoU.expand(b,c,h,w)
    alpha = 0.75
    gamma = 2
    num_classes = 2
    # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True

    # BCE_loss = BCE_loss*targets

    # at = alpha_f.gather(0, targets.data.view(-1))

    pt = torch.exp(-BCE_loss)
    # pt = inputs*targets+(1-inputs)*(1-targets)
    # print((pt.mean()))

    # print((BCE_loss.mean()))
    F_loss =  torch.mul(((1-pt) ** gamma) ,BCE_loss)
    # print((F_loss.mean()))
    # F_loss1 = torch.mul(F_loss,targets) #计算正样本loss
    # F_loss2 = torch.mul(F_loss , (1-targets))  # 计算正样本loss

    # targets = targets.type(torch.long)
    at = targets*alpha+(1-targets)*(1-alpha)
    # F_loss = F_loss.view(-1)
    # F_loss = torch.mul(F_loss,at)
    # IoU = IoU.view(-1)

    # at = targets * alpha + (1 - targets) * (1 - alpha)


    F_loss = (1-IoU)*(F_loss)**(IoU*0.5+0.5)
    # F_loss = at * F_loss
    F_loss_map = at * F_loss


    F_loss_sum = F_loss_map.sum()
    # print((F_loss.mean()))
    return F_loss_map,F_loss_sum



def SoftIoULoss(pred, target):
        pred = torch.nn.Sigmoid()(pred)
        # print("pred.shape: ", pred.shape)
        # pred = 0.999*(pred - 0.5)+0.5

        intersection = torch.mul(pred,target)

        # intersection_sum = intersection.sum()
        smooth = 1

        loss = (intersection.sum() +smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
        # loss = (2*intersection.sum() + smooth) / (pred.sum() + target.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss



def FocalLoss(inputs, targets):
    "Non weighted version of Focal Loss"

    # def __init__(self, alpha=.25, gamma=2):
    #     super(WeightedFocalLoss, self).__init__()

    alpha = 0.75
    gamma = 2
    num_classes = 2

    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True


    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    targets = targets.type(torch.long)

    # at = alpha_f.gather(0, targets.data.view(-1))
    at = targets*alpha+(1-targets)*(1-alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss

    F_loss = at * F_loss
    return F_loss.sum()











class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    predict = (output > 0).float()  # P
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1) # T
    elif len(target.shape) == 4:
        target = target.float()  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())  # TP

    # areas of intersection and union
    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter
    # print('area_inter:', area_inter)
    # print('area_pred:', area_pred)
    # print('area_lab:', area_lab)
    # print('area_union:', area_union)
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union