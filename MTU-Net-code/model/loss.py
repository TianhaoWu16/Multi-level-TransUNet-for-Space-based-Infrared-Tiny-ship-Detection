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

    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)
    
    alpha = 0.75
    gamma = 2
    num_classes = 2
    # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True

 
    pt = torch.exp(-BCE_loss)
    
    F_loss =  torch.mul(((1-pt) ** gamma) ,BCE_loss)
   
    at = targets*alpha+(1-targets)*(1-alpha)
    
    F_loss = (1-IoU)*(F_loss)**(IoU*0.5+0.5)
   
    F_loss_map = at * F_loss


    F_loss_sum = F_loss_map.sum()
    
    return F_loss_map,F_loss_sum



def SoftIoULoss(pred, target):
        pred = torch.nn.Sigmoid()(pred)
       

        intersection = torch.mul(pred,target)

       
        smooth = 1

        loss = (intersection.sum() +smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
       
        loss = 1 - loss.mean()

        return loss



def FocalLoss(inputs, targets):

    alpha = 0.75
    gamma = 2
    num_classes = 2

  
    gamma = gamma
    size_average = True


    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    targets = targets.type(torch.long)

    
    at = targets*alpha+(1-targets)*(1-alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss

    F_loss = at * F_loss
    return F_loss.sum()



