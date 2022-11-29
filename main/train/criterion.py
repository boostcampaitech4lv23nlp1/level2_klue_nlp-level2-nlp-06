import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module): 
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        

    def forward(self, inputs, targets):
        targets = targets.type(torch.long)
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs.float(), targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return torch.mean(F_loss)

