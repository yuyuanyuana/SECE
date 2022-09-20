
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F

def nb_loss(data, mean, disp):
    eps = 1e-10
    loss1 = torch.lgamma(disp+eps) + torch.lgamma(data+1) - torch.lgamma(data+disp+eps)
    loss2 = (disp+data) * torch.log(1.0 + (mean/(disp+eps))) + (data * (torch.log(disp+eps) - torch.log(mean+eps)))
    return loss1 + loss2


def zinb_loss(data, mean, disp, drop, ridge_lambda=0):
    eps = 1e-10
    nb_case = nb_loss(data, mean, disp) - torch.log(1.0-drop+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case =  -torch.log(drop + ((1.0-drop)*zero_nb)+eps)
    result = torch.where(torch.lt(data, 1e-10), zero_case, nb_case)
    ridge = ridge_lambda*torch.pow(drop, 2)
    result += ridge
    return result.mean()


def nll_loss(data, mean, disp, drop=None, dist='zinb'):
    if dist == 'nb':
        return nb_loss(data, mean, disp).mean()
    else:
        return zinb_loss(data, mean, disp, drop, ridge_lambda=0)


def plot_loss2(loss_list, result_path, labels = ['loss', 'loss_re', 'loss_kernal']):
    train_loss = np.array(loss_list)
    for i in range(train_loss.shape[1]):
        plt.plot(train_loss[:,i], label=labels[i])
    plt.legend()
    plt.savefig(f'{result_path}/model2_loss.png')
    plt.close()


