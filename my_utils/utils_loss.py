import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def forward_loss(f, K, labels, device):
    Q = torch.ones(K,K) * 1/(K-1)
    Q = Q.to(device)
    for k in range(K):
        Q[k,k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long())


def pc_loss(f, K, labels, device):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss



def c_loss(output, label, K, device):
    loss = nn.MSELoss(reduction='mean')
    one_hot = F.one_hot(label.to(torch.int64), K) * 2 - 1
    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size())
    y_label = y_label.to(device)
    output = loss(sig_out, y_label)
    return output


def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def ce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1-(pow_outputs*Y).sum(dim=1))/q # n
    return sample_loss


def c_TF_loss(output_F, target_SL_F, K, device):
    target_SL_F = target_SL_F.T
    A_num = target_SL_F.size()[0]
    loss = 0
    for i in range(K):
        label_i = torch.zeros(target_SL_F.size()[1]) + i
        loss += c_loss(output_F, label_i, K, device)
    for i in range(A_num):
        loss -= c_loss(output_F, target_SL_F[i], K, device)
    return loss


def calculate_loss(output, target_TF, target_SL, data_prior, K, criterion, loss_fn):
    device = output.device
    index_T = []
    index_F = []
    for i in range(target_TF.size()[0]):
        if target_TF[i] == K:
            index_F.append(i)
        else:
            index_T.append(i)

    index_F = torch.tensor(index_F).long()
    index_T = torch.tensor(index_T).long()
    index_T = index_T.to(device)
    index_F = index_F.to(device)
    target_TL = torch.index_select(target_TF, dim=0, index=index_T)
    output_TL = torch.index_select(output, dim=0, index=index_T)
    target_SL = torch.index_select(target_SL, dim=0, index=index_F)
    output_SL = torch.index_select(output, dim=0, index=index_F)

    partialY = 1 - target_SL
    can_num = partialY.sum(dim=1).float()  # n
    can_num = 1.0 / can_num

    # target_TL_loss = c_loss(output_TL, target_TL, K, device)
    if loss_fn == 'ce':
        target_TL_loss = criterion(output_TL, target_TL)
        target_SL_loss = can_num * ce_loss(output_SL, partialY)
    elif loss_fn == 'mse':
        target_TL_loss = c_loss(output_TL, target_TL, K, device)
        target_SL_loss = can_num * mse_loss(output_SL, partialY)
    elif loss_fn == 'mae':
        loss = nn.L1Loss(reduction='mean')
        _, predicted = torch.max(output_TL, 1)
        target_TL_loss = loss(predicted.float(), target_TL.float())
        target_SL_loss = can_num * mae_loss(output_SL, partialY)
    elif loss_fn == 'gce':
        target_TL_loss = criterion(output_TL, target_TL)
        target_SL_loss = can_num * gce_loss(output_SL, partialY)

    target_SL_loss = target_SL_loss.sum() / partialY.shape[0]

    loss = (target_TL_loss) + target_SL_loss
    return loss