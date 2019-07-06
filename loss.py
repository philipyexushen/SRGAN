import torch
from torch import nn
from torchvision.models.vgg import vgg16
from numba import jit

class GeneratorLoss(nn.Module):
    def __int__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        pass


class TVLoss(nn.Module):
    """
    Total variation loss实现
    """
    def __init__(self, tv_loss_weight: float = 1):
        super(TVLoss, self).__init__()
        self.__tv_loss_weight = tv_loss_weight

    def forward(self, x):
        @jit
        def tensor_size(t)->torch.Tensor:
            return t.size()[1] * t.size()[2] * t.size()[3]

        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = tensor_size(x[:, :, 1:, :])
        count_w = tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.__tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
