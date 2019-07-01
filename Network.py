import math
import torch.nn.functional as F
from torch import nn

class ResidualBlock(nn.Module):
    """
     论文里面的那个残差网络结构
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.__conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.__bn1 = nn.BatchNorm2d(channels)
        self.__prelu = nn.PReLU()
        self.__conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.__bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.__conv1(x)
        residual = self.__bn1(residual)
        residual = self.__prelu(residual)
        residual = self.__conv2(residual)
        residual = self.__bn2(residual)

        return x + residual


class UpSampleBlock(nn.Module):
    def __init__(self, channels, up_scale):
        super(UpSampleBlock, self).__init__()
        self.__conv = nn.Conv2d(channels, channels * up_scale ** 2, kernel_size=3, padding=1)
        self.__pixel_shuffler = nn.PixelShuffle(up_scale)
        self.__prelu = nn.PReLU()


    def forward(self, x):
        x = self.__conv(x)
        x = self.__pixel_shuffler(x)
        x = self.__prelu(x)
        return x


class GeneratorNet(nn.Module):
    def __init__(self, scale_factor):
        up_sample_block_sum = int(math.log(scale_factor, 2))

        super(GeneratorNet, self).__init__()
        self.__block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU()
        )

        self.__block2 = ResidualBlock(64)
        self.__block3 = ResidualBlock(64)
        self.__block4 = ResidualBlock(64)
        self.__block5 = ResidualBlock(64)
        self.__block6 = ResidualBlock(64)

        self.__block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.__block8 = nn.ModuleList([UpSampleBlock(64, 2) for _ in range(up_sample_block_sum)])
        self.__block8.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.__block8 = nn.Sequential(*self.__block8)

    def forward(self, x):
        x = self.__block1(x)
        x = self.__block2(x)
        x = self.__block3(x)
        x = self.__block4(x)
        x = self.__block5(x)
        x = self.__block6(x)
        x = self.__block7(x)
        x = self.__block8(x)

        return (F.tanh(x) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.__block: nn.Module = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 下面两步相当于Dense了
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.__block(x).view(batch_size))
